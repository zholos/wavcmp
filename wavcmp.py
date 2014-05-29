#!/usr/bin/env python
from __future__ import print_function

import bisect, sys, os, tempfile, subprocess, json, warnings, argparse
import scipy.io.wavfile
import numpy as np


class SilenceableError(RuntimeError):
    pass


class File:
    """Similar to dedup.File, this represents a file (named on the command
    line), which may or may not be a valid Track.
    """

    def __init__(self, filename):
        self.filename = filename

        process = subprocess.Popen(
            ["ffprobe", "-v", "quiet", "-i", filename,
                "-print_format", "json", "-show_streams", "-show_error"],
            stdin=open(os.devnull, "r"), stdout=subprocess.PIPE)
        out, _ = process.communicate()
        ret = process.wait()
        # Assume that if ffprobe worked correctly, we get a valid JSON object
        # with either "streams" (valid Track) or "error" (keep as File).
        try:
            probe = json.loads(out)
            if ret if "streams" in probe else "error" not in probe:
                raise ValueError
        except ValueError:
            raise RuntimeError("ffprobe failed on file: '{}'".format(filename))

        # To qualify as a track, the file must have one audio stream with
        # 2 channels and no video streams.
        rate = None
        for s in probe.get("streams") or ():
            if s.get("codec_type") == "audio":
                if s.get("channels") != 2:
                    return
                if rate is not None:
                    return
                rate = int(s["sample_rate"])
                tsn, tsd = map(int, s["time_base"].split("/"))
                duration = int(s["duration_ts"]) * rate * tsn // tsd
                # remainder discarded, this is inaccurate anyway
            elif s.get("codec_type") == "video":
                if s.get("disposition", {}).get("attached_pic"): # cover art
                    pass
                else:
                    return
        if rate is None:
            return

        self.__class__ = Track
        self.rate = rate
        self.duration = duration # not "size", could mean duration * channels

class Track(File):
    # Probed duration may be inaccurate.
    duration_accuracy = 5 # +/- seconds

    def _load_data(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
            subprocess.check_call(
                ["ffmpeg", "-v", "quiet", "-i", self.filename,
                    "-f", "wav", "-y", temp.name],
                stdin=open(os.devnull, "r"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # WavFileWarning because "fmt" chunk is size 18,
                # as ffmpeg always outputs it
                rate, data = scipy.io.wavfile.read(temp.name)

        assert data.ndim == 2
        if rate != self.rate or data.shape[1] != 2 or \
                abs(data.shape[0] - self.duration) > \
                self.duration_accuracy * rate:
            raise RuntimeError(
                "Data didn't match probe on file: '{}'".format(self.filename))
        assert data.dtype == np.int16
        self._data = data

    def data(self):
        while True:
            try:
                return self._data
            except AttributeError:
                self._load_data()

    data_high = -np.iinfo(np.int16).min

    def data_wider(self):
        # The following operations are required not to overflow:
        # - subtracting one sample from another (a-b)
        # - summing a small number of consecutive samples (group_sums), and
        #   subtracting one sum from another
        # - summing differences over entire track together
        # Converting to a larger type avoids overflow in the first two cases;
        # the third case requires a different type.
        # "F" layout better for operations on specific channels.
        return np.asarray(self.data(), dtype=np.int32, order="F")


def _small(n, d, digits=7, sign=False):
    assert d > 0 or n == d == 0
    if n == 0:
        return "0"
    elif sign:
        return ("+" if n > 0 else "-") + _small(abs(n), d, digits=digits)
    else:
        assert n > 0
        m = pow(10, digits)
        i = (100 * m * n + d // 2) // d # round .5 up
        return str(i // m) + "." + str(i % m).zfill(digits) + "%"

def _duration(samples, rate):
    assert samples >= 0
    s = samples // rate
    f = "{2}" if s < 60 else "{1}:{2:02}" if s < 3600 else "{0}:{1:02}:{2:02}"
    f += ".{3:03}"
    return f.format(s // 3600, s // 60 % 60, s % 60,
                    samples % rate * 1000 // rate)

def _sum(a):
    # int64 should be sufficient for int32 data
    # convert to long int to avoid overflow later
    return int(np.sum(a, dtype=np.int64))

def _spectrum(a):
    a = np.mean(a, axis=1, dtype=np.float_) / Track.data_high
    window = 8192 # should be a power of 2
    a.resize(-(-len(a)//window)*window) # pad with zeroes
    s = np.fft.fft(a.reshape((-1, window)))
    s = np.mean(np.abs(s[...,:window//2])**2, axis=0)/window**2
    s[1:] *= 2 # combine positive and negative frequencies (skip 0)
    # Audacity further multiplies this by 2 to normalize power, and implicitly
    # by another 4 because the input signal is taken as the sum of two channels
    return s

class Segment:
    """For display purposes, the two tracks are split into segments, which can
    either be a common segment or a padding segment on either end. Statistics
    for both types of segment are similar in calculation.
    """

    def __init__(self, ac, bc, rate, total, padding=None):
        assert ac.shape == bc.shape
        self.ac = ac
        self.bc = bc
        self.rate = rate
        self.total = total
        self.padding = padding
        assert padding in (None, "-", "+")

    def classify(self):
        if not np.any(self.ac) and not np.any(self.bc):
            return "silence"
        elif np.all(self.ac == self.bc):
            return "identical"

    def ds(self):
        """Computes difference metric."""
        return _sum(np.abs(self.ac - self.bc))

    def ds_str(self):
        return _small(self.ds(), self.total * Track.data_high)

    def zs(self):
        """Computes number of different samples."""
        return _sum(self.ac != self.bc)

    def zs_str(self):
        return _small(self.zs(), self.total, digits=5)

    def share_str(self):
        """Computes a measure of contribution to discrepancies by one or the
        other track."""
        assert not self.padding
        i = self.ac != self.bc
        aci = self.ac[i]
        bci = self.bc[i]
        sn = _sum(np.abs(bci)-np.abs(aci))
        sd = _sum(np.abs(bci)+np.abs(aci))
        return _small(sn, sd, sign=True)

    def cutoff_str(self):
        assert not self.padding
        asc = np.cumsum(_spectrum(self.ac)[::-1])[::-1]
        bsc = np.cumsum(_spectrum(self.bc)[::-1])[::-1]
        db = 10*(np.log10(bsc)-np.log10(asc))
        gain = np.argmax(db)
        drop = np.argmin(db)
        i = drop if -db[drop] > db[gain] else gain
        freq = self.rate * max(0, i-.5) / (2.*len(db))
        return "{:+.1f} dB >{:.1f} kHz".format(db[i], freq/1000)

    def duration_str(self):
        return _duration(len(self.ac), self.rate)

    def format(self, verbose=False):
        if verbose:
            f = "  {0}{1} ({2} samples)"
        elif self.padding:
            f = "{0}{1} ({2})"
        else:
            f = "{0}{1}"
        s = f.format(self.padding or "", self.duration_str(), len(self.ac))
        classify = verbose and self.classify()
        if classify:
            s += ", {}".format(classify)
        else:
            if verbose:
                f = ", {0} MAD, {1} non-zero"
            else:
                f = ", {0} {1}"
            s += f.format(self.ds_str(), self.zs_str())
            if verbose and not self.padding:
                f = ", {0} share, {1}"
                s += f.format(self.share_str(), self.cutoff_str())
        return s

class Match:
    """Match details."""

    def __init__(self, a, b, offset):
        self.a = a
        self.b = b
        self.offset = offset

    def segments(self):
        a = self.a.data_wider()
        b = self.b.data_wider()
        offset = self.offset
        acs = np.split(a, [max(0, offset), len(b)+offset])
        bcs = np.split(b, [max(0, -offset), len(a)-offset])
        for i, (ac, bc) in enumerate(zip(acs, bcs)):
            if i == 1:
                assert len(ac) == len(bc)
                yield Segment(ac, bc, self.a.rate,
                              min(a.size, b.size)) # matches total in cmp_track
            elif len(ac):
                assert not len(bc)
                # careful with np.zeros type
                yield Segment(ac, ac*0, self.a.rate, ac.size, padding="-")
            elif len(bc):
                yield Segment(bc*0, bc, self.a.rate, bc.size, padding="+")

    def common(self):
        for segment in self.segments():
            if not segment.padding:
                return segment

    def ds(self):
        return self.common().ds()

    def show_machine_readable(self):
        s = self.common()
        print(self.offset, s.ds_str(), s.zs_str())

    def show(self, verbose=False):
        if verbose:
            print(self.a.filename, "~", self.b.filename)
        print(*(s.format(verbose=verbose) for s in self.segments()),
              sep=("\n" if verbose else " | "))


def _group_sums(a, group):
    a = a[:len(a)//group*group]
    return np.sum(a.reshape((len(a)//group, group) + a.shape[1:]), axis=1,
                  dtype=a.dtype) # dtype defaults to int64 on 64-bit machine

def _limited_ds(ac, bc, limit):
    assert len(ac) == len(bc)
    step = 1<<13 # chosen empirically
    s = 0

    # iterate over chunks pseudo-randomly so less likely to keep summing silence
    off = xrange(0, len(ac), step)
    i = 0
    a = len(off) * 4 + 1 # Full-period theorem
    c = 2305843009213693951 # Mersenne prime
    assert len(off) % c # choose a larger prime c if this limit is ever reached
    for _ in off:
        i = (a * i + c) % len(off)
        ax = ac[off[i]:off[i]+step]
        bx = bc[off[i]:off[i]+step]
        s += _sum(np.abs(ax-bx))
        if s > limit:
            return
    return s

def _cmp_right(a, b, max_offset, matches):
    """Compare only for positive offsets (delaying b relative to a)."""

    # Create a shorter series by summing small sequences of consecutive samples.
    # Since |a+b|<=|a|+|b|, the metric over sums is a lower bound on the metric
    # over the original series, so we can reject many offsets earlier.

    # The groups must line up between a and the delayed b, so the groups for b
    # are recalculated at different shifts, e.g. shift=1:
    #   a [-|-----|-----|-----|-...-|----]
    #   b   [-----|-----|-----|-...-|--]

    def limit():
        return matches[0][0] * 2

    group = 7 # best value probably depends on cache size and track frequencies
    bg = _group_sums(b, group)

    for shift in xrange(group):
        ag = _group_sums(a[shift:], group)

        for offset in xrange(shift, max_offset+1, group):
            # TODO: adjust range to conditions instead of checking every loop
            if offset > len(a): # include test at zero overlap for completeness
                continue
            if abs(offset + len(b) - len(a)) > max_offset:
                continue

            # *c is common (overlapping) part
            agc = ag[offset//group:][:len(bg)]
            bgc = bg[:len(ag)-offset//group]
            dsg = _limited_ds(agc, bgc, limit())
            if dsg is not None:
                ac = a[offset:][:len(b)]
                bc = b[:len(a)-offset]
                ds = _limited_ds(ac, bc, limit())
                if ds is not None:
                    bisect.insort(matches, (ds, offset))
                    while matches[-1][0] > limit(): # make sure to keep ds=0
                        matches.pop()

def cmp_track(a, b, offset=None, threshold=None):
    """Compare two tracks at different offsets and yields good matches.

    The difference metric is the sum of absolute differences (SAD) over the
    common part. A match is a particular offset at which this metric is below a
    certain absolute threshold. There may be multiple good matches. The first
    returned match has the lowest metric, the rest are ordered and their metrics
    are not more than double that of the first match.

    The metric and return conditions are chosen this way to quickly reject non-
    matches.
    """

    assert a.rate == b.rate

    if offset is None:
        offset = 5
    max_offset = -int(-a.rate*offset)

    # Offset basically means ignored padding at the front in one of the tracks,
    # also limits padding at the back.
    if abs(a.duration - b.duration) > \
            2 * max_offset + 2 * a.duration_accuracy * a.rate:
        return

    ax = a.data_wider()
    bx = b.data_wider()

    if threshold is None:
        # -V 1 MP3 should just clear 1%, -b 320 should also
        threshold = 0.01
    total = min(ax.size, bx.size) # fixed denominator regardless of overlap
    limit = int(a.data_high * total * threshold) # absolute threshold

    matches = [(limit // 2, None)] # [pairs of (SAD, offset)]
    _cmp_right(bx, ax, max_offset, matches)
    # mirror; remove 0 offset from first call, second call will add it again
    matches = [(ds, -offset if offset else None) for ds, offset in matches]
    _cmp_right(ax, bx, max_offset, matches)
    matches = ((ds, offset) for ds, offset in matches if offset is not None)
    # sort for ordered offset for equal MAD, offset=0 is best
    matches = sorted(matches, key=lambda x: (x[0], abs(x[1]), x[1]<0))
    for ds, offset in matches:
        match = Match(a, b, offset)
        assert ds == match.ds()
        yield match


def main():
    parser = argparse.ArgumentParser(
        prog="wavcmp",
        description="Compare two tracks at different offsets. Return true if "
                    "they match.",
        add_help=False)

    parser.add_argument("-h", "--help", action="help", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version="%(prog)s 0.1",
                        help=argparse.SUPPRESS)
    parser.add_argument("-o", metavar="offset", type=float,
                        help="maximum offset, default 0.5 seconds")
    parser.add_argument("-t", metavar="threshold", type=float,
                        help="match threshold, default 1%%")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", action="store_true",
                       help="verbose match display")
    group.add_argument("-q", action="store_true",
                       help="don't show match statistics")
    group.add_argument("-M", action="store_true",
                        help="machine-readable output")
    parser.add_argument("-s", action="store_true",
                        help="silently fail on invalid files")
    #parser.add_argument("-w", metavar="write",
    #                    help="write difference for visual inspection")
    parser.add_argument("a", metavar="a.flac") # help=argparse.SUPPRESS
    parser.add_argument("b", metavar="b.flac") # help=argparse.SUPPRESS
    args = parser.parse_args()

    try:
        a, b = map(File, (args.a, args.b))
        for i in (a, b):
            if not isinstance(i, Track):
                raise SilenceableError(
                    "Not a stereo audio file: "
                    "'{}'".format(i.filename))
        if a.rate != b.rate:
            raise SilenceableError(
                "Sample rates different in files: "
                "'{}' and '{}'".format(a.filename, b.filename))
        matches = cmp_track(a, b, offset=args.o,
                            threshold=None if args.t is None else args.t/100.)

        matched = False
        for match in matches:
            matched = True
            if args.q:
                break
            elif args.M:
                match.show_machine_readable()
                break
            else:
                match.show(verbose=args.v)
        return matched

    except SilenceableError:
        if args.s:
            return False
        else:
            raise

if __name__ == "__main__":
    try:
        sys.exit(int(not main()))
    except (EnvironmentError, RuntimeError) as e:
        print("{0}: error: {1}".format(os.path.basename(sys.argv[0]), e),
              file=sys.stderr)
        sys.exit(1)

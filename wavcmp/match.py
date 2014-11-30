from __future__ import print_function

import warnings
import numpy as np

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

def _spectrum(a, high):
    a = np.mean(a, axis=1, dtype=np.float_) / high
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

    def __init__(self, ac, bc, high, rate, total, padding=None):
        assert ac.shape == bc.shape
        self.ac = ac
        self.bc = bc
        self.high = high
        self.rate = rate
        self.total = total
        self.padding = padding
        assert padding in (None, "-", "+", "=", "<", ">")

    def classify(self):
        if not np.any(self.ac) and not np.any(self.bc):
            return "silence"
        elif np.all(self.ac == self.bc):
            return "identical"

    def ds(self):
        """Computes difference metric."""
        return _sum(np.abs(self.ac - self.bc))

    def ds_str(self):
        return _small(self.ds(), self.total * self.high)

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
        asc = np.cumsum(_spectrum(self.ac, self.high)[::-1])[::-1]
        bsc = np.cumsum(_spectrum(self.bc, self.high)[::-1])[::-1]
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
        elif self.padding in ("-", "+", "="):
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

class _Result:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def show(self, verbose=False):
        if verbose:
            print(self.a.title(), "~", self.b.title())
        print(*(s.format(verbose=verbose) for s in self.segments()),
              sep=("\n" if verbose else " | "))

class Match(_Result):
    """Match details."""

    def __init__(self, a, b, offset):
        assert (a or b) and (a and b or offset == 0)
        _Result.__init__(self, a, b)
        self.offset = offset

    def end_offset(self):
        if self.a and self.b:
            # matches comparison with max_offset in _cmp_right
            return self.offset + len(self.b.data()) - len(self.a.data())
        else:
            return 0

    def segments(self):
        if self.a and self.b:
            assert self.a.data_high == self.b.data_high
            assert self.a.rate == self.b.rate
            a = self.a.data_wider()
            b = self.b.data_wider()
            offset = self.offset
            with warnings.catch_warnings():
                # Changing handling of empty arrays not relevant to us
                warnings.simplefilter("ignore", FutureWarning)
                acs = np.split(a, (max(0, offset), len(b)+offset))
                bcs = np.split(b, (max(0, -offset), len(a)-offset))
            for i, (ac, bc) in enumerate(zip(acs, bcs)):
                if i == 1:
                    assert len(ac) == len(bc)
                    yield Segment(
                        ac, bc, self.a.data_high, self.a.rate,
                        min(a.size, b.size)) # matches total in cmp_track
                elif len(ac):
                    assert not len(bc)
                    # careful with np.zeros type
                    yield Segment(ac, ac*0, self.a.data_high, self.a.rate,
                                  ac.size, padding="-")
                elif len(bc):
                    yield Segment(bc*0, bc, self.a.data_high, self.a.rate,
                                  bc.size, padding="+")
        elif self.a:
            ac = self.a.data_wider()
            yield Segment(ac, ac*0, self.a.data_high, self.a.rate,
                          ac.size, padding="<")
        elif self.b:
            bc = self.b.data_wider()
            yield Segment(bc*0, bc, self.b.data_high, self.b.rate,
                          bc.size, padding=">")

    def common(self):
        for segment in self.segments():
            if not segment.padding:
                return segment

    def ds(self):
        return self.common().ds()

    def show_machine_readable(self):
        s = self.common()
        print(self.offset, s.ds_str(), s.zs_str())

class MatchSequence(_Result):
    """Compound match from a sequence of matching tracks, such as an album."""

    def __init__(self, a, b, sequence):
        _Result.__init__(self, a, b)
        self.sequence = list(sequence)

    def segments(self):
        last = None
        for match in self.sequence:
            for segment in match.segments():
                if last:
                    if {last.padding, segment.padding} == {"-", "+"} and \
                            len(last.ac) == len(segment.ac):
                        assert last.high == segment.high
                        assert last.rate == segment.rate
                        assert last.total == segment.total # equal to padding
                        yield Segment(last.ac + segment.ac,
                                      last.bc + segment.bc,
                                      last.high, last.rate, last.total,
                                      padding="=") # better legibility than "*"
                        segment = None
                    else:
                        yield last
                last = segment
        if last:
            yield last

    def ds(self):
        ds = 0
        for segment in self.segments():
            if not segment.padding or segment.padding == "=":
                ds += segment.ds()
        return ds

    def show_machine_readable(self):
        raise RuntimeError("-M output not defined for directories")

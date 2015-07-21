import os, os.path, subprocess, struct, json, warnings
import numpy as np

from .cmp import cmp_track, cmp_album

class SilenceableError(RuntimeError):
    pass

class File:
    """Similar to dedup.File, this represents a file (named on the command
    line), which may or may not be a valid Track.
    """

    def __init__(self, filename):
        self.filename = filename
        for kind in (Album, Track):
            kind._probe(self)
            if isinstance(self, kind):
                break

class Audio(File):
    def title(self):
        title = self.filename
        if self.bps:
            title += " [{:.0f} kbps]".format(self.bps / 1000.)
        return title

class Track(Audio):
    @staticmethod
    def _ffprobe(filename, sections, args):
        process = subprocess.Popen(
            ["ffprobe", "-v", "quiet", "-i", filename, "-print_format", "json",
                "-show_error"] + ["-show_"+s for s in sections] + args,
            stdin=open(os.devnull, "r"), stdout=subprocess.PIPE, bufsize=-1)
        out, _ = process.communicate()
        ret = process.wait()
        # Assume that if ffprobe worked correctly, we get a valid JSON object
        # with either the requested section (valid Track) or "error" (keep as
        # File).
        try:
            probe = json.loads(out.decode("ascii", "replace"))
        except ValueError:
            probe = {} # RuntimeError later
        if "error" in probe: # ret != 0 in this case
            return # no RuntimeError, but will be rejected by _probe
        if ret == 0 and all(s in probe for s in sections):
            return probe
        else:
            raise RuntimeError("ffprobe failed on file: '{}'".format(filename))

    @staticmethod
    def _probe(self):
        # To qualify as a track, the file must have one audio stream with
        # 2 channels and no video streams.
        rate = None
        probe = Track._ffprobe(
            self.filename, ["streams", "format"],
            ["-show_entries", "format=probe_score:format_tags="])
            # tag encoding may break JSON parser
        if not probe:
            return
        if probe["format"]["probe_score"] <= 25:
            # This score triggers a misdetection warning from ffprobe
            return
        for s in probe["streams"]:
            if s.get("codec_type") == "audio":
                if s["channels"] != 2:
                    return
                if rate is not None: # second audio stream
                    return
                rate = int(s["sample_rate"])
                index = int(s["index"])
                duration = s.get("duration_ts")
                tsn, tsd = map(int, s["time_base"].split("/"))
                bps = None
                if s["codec_name"] == "mp3":
                    try:
                        bps = int(s.get("bit_rate"))
                    except ValueError:
                        pass
            elif s.get("codec_type") == "video":
                if s.get("disposition", {}).get("attached_pic"): # cover art
                    pass
                else:
                    return
        if rate is None:
            return

        # Separate from probing to reject non-tracks as quickly as possible.
        # Slower than estimate above, but still much faster than reading data.
        probe = Track._ffprobe(
            self.filename, ["packets"],
            ["-select_streams", str(index), "-show_entries", "packet=duration"])
        if not probe:
            return
        packets = probe["packets"]
        if packets and "duration" not in packets[0]:
            pass # no duration data for e.g. Monkey's Audio; keep header value
        else:
            duration = sum(int(p["duration"]) for p in packets)
        duration = duration * rate * tsn // tsd
        # remainder discarded, this is not precise anyway

        self.__class__ = Track
        self.rate = rate
        self.duration = duration # not "size", could mean duration * channels
        self.bps = bps

    # Probed duration may be inaccurate (even when counting packets).
    duration_accuracy = 1 # +/- seconds

    def _read_data(self):
        out = subprocess.check_output(
            ["ffmpeg", "-v", "quiet", "-i", self.filename,
                    "-f", "au", "-acodec", "pcm_s16be", "pipe:"],
            stdin=open(os.devnull, "r"), bufsize=-1)
        magic, offset, _, codec, rate, channels = struct.unpack(">6I", out[:24])
        assert magic == 0x2e736e64 and codec == 3 and offset > 24
        data = np.frombuffer(out, offset=offset, dtype=">i2")
        data = data.reshape((-1, channels)).astype(np.int16, order="F")
        if rate != self.rate or data.shape[1] != 2 or \
                abs(data.shape[0] - self.duration) > \
                self.duration_accuracy * rate:
            raise RuntimeError(
                "Data didn't match probe on file: '{}'".format(self.filename))
        return data

    def data(self):
        if not hasattr(self, "_data"):
            self._data = self._read_data()
        return self._data

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
        return self.data().astype(np.int32, order="F")

    def cmp(self, other, **options):
        if not isinstance(other, Track):
            raise SilenceableError(
                "Can't compare file and directory: "
                "'{}' and '{}'".format(self.filename, other.filename))
        return cmp_track(self, other, **options)

class Album(Audio):
    @staticmethod
    def _probe(self):
        if not os.path.isdir(self.filename):
            return
        tracks = []
        for item in sorted(os.listdir(self.filename)):
            node = File(os.path.join(self.filename, item))
            if isinstance(node, (Track, Album)):
                if tracks and tracks[-1].rate != node.rate:
                    raise SilenceableError(
                        "Sample rates different in files: "
                        "'{}' and '{}'".format(
                            tracks[-1].filename, node.filename))
            if isinstance(node, Track):
                tracks.append(node)
            elif isinstance(node, Album):
                tracks += node.tracks
        if not tracks:
            return

        self.__class__ = Album
        self.tracks = tracks
        self.rate = tracks[0].rate
        if any(i.bps is None for i in tracks):
            self.bps = None
        else:
            # estimate for display
            self.bps = sum(i.bps * i.duration for i in tracks) / \
                       sum(i.duration for i in tracks)

    def cmp(self, other, **options):
        if not isinstance(other, Album):
            raise SilenceableError(
                "Can't compare directory and file: "
                "'{}' and '{}'".format(self.filename, other.filename))
        return cmp_album(self, other, **options)

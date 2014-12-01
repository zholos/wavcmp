__all__ = ["match", "cmp", "track", "cmdline"]
__version__ = "0.4"

from .match import Segment, Match, MatchSequence
from .cmp import _algorithm, cmp_track, cmp_album
from .track import SilenceableError, File, Audio, Track, Album

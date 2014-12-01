import bisect, itertools, multiprocessing.dummy, contextlib
import numpy as np

from .match import _sum, Match, MatchSequence

try:
    xrange
except NameError:
    xrange = range

def _group_sums(a, group):
    assert a.ndim == 1
    a = a[:len(a)//group*group]
    return np.sum(a.reshape((-1, group)), axis=1,
                  dtype=a.dtype) # dtype defaults to int64 on 64-bit machine

def _limited_ds(ac, bc, limit):
    assert len(ac) == len(bc)
    # iterate over chunks pseudo-randomly so less likely to keep summing silence
    step = 1<<13 # chosen empirically
    n = -(-len(ac) // step) or 1
    a = n * 4 + 1 # Full-period theorem
    c = 2305843009213693951 # Mersenne prime
    assert n % c # choose a larger prime c if this limit is ever reached
    s = 0
    i = 0
    while True:
        i = (a * i + c) % n
        off = i * step
        ax = ac[off:off+step]
        bx = bc[off:off+step]
        s += _sum(np.abs(ax-bx))
        if s > limit:
            return
        if i == 0:
            break
    return s

def _cmp_candidates(ag, bg, start, stop, limit, check_match):
    for i in xrange(start, stop):
        # *c is common (overlapping) part
        a0 = max(0, i)
        b0 = max(0, -i)
        agc = ag[a0:][:len(bg)-b0]
        bgc = bg[b0:][:len(ag)-a0]
        dsg = _limited_ds(agc, bgc, limit)
        if dsg is not None:
            limit = check_match(i) # simpler than yield

_algorithm = "Python"
_use_threads = False

try:
    from . import _compiled # optimized Cython routines
except ImportError:
    pass
else:
    _limited_ds = _compiled.limited_ds
    _cmp_candidates = _compiled.cmp_candidates
    _algorithm = _compiled.algorithm
    _use_threads = True

def cmp_track(a, b, offset=None, threshold=None, skip=None, threads=None):
    """Compare two tracks at different offsets and yields good matches.

    The difference metric is the sum of absolute differences (SAD) over the
    common part. A match is a particular offset at which this metric is below a
    certain absolute threshold. There may be multiple good matches. The first
    returned match has the lowest metric, the rest are ordered and their metrics
    are not more than double that of the first match.

    The metric and return conditions are chosen this way to quickly reject non-
    matches.
    """

    if offset is None:
        offset = 5
    if threshold is None:
        # -V 1 MP3 should just clear 1%, -b 320 should also
        threshold = 0.01
    assert offset >= 0 and threshold >= 0

    assert a.rate == b.rate
    offset_bound = -int(-a.rate*offset)

    # Offset basically means ignored padding at the front in one of the tracks,
    # also limits padding at the back.
    if abs(a.duration - b.duration) > \
            2 * offset_bound + 2 * a.duration_accuracy * a.rate:
        return

    ax = a.data_wider()
    bx = b.data_wider()
    total = min(ax.size, bx.size) # fixed denominator regardless of overlap
    limit = [int(a.data_high * total * threshold)] # absolute threshold
    matches = [] # [(SAD, offset)], ordered
    mutex = multiprocessing.dummy.Lock()

    # Exact check, called for only a few candidates by heuristic algorithm.
    def check_match(offset):
        assert -len(bx) <= offset <= len(ax)
        ac = ax[max(0, offset):][:len(bx)-max(0, -offset)]
        bc = bx[max(0, -offset):][:len(ax)-max(0, offset)]
        ds = _limited_ds(ac, bc, limit[0])
        if ds is not None:
            with mutex:
                bisect.insort(matches, (ds, offset))
                limit[0] = matches[0][0] * 2 # assuming assign is atomic
                while matches[-1][0] > limit[0]: # make sure to keep ds=0
                    matches.pop()
        return limit[0]

    # Match offset is delay of b start from a start:
    #   a [-----------]
    #   b (offset)[-----]

    # An overlap of exactly zero is included for completeness.
    # Range of possible offsets derived from these conditions:
    #   -len(b) <= offset <= len(a)
    #   abs(offset) <= offset_bound
    #   abs(offset+len(bx)-len(ax)) <= offset_bound
    min_offset = max(-len(bx), -offset_bound + max(0, len(ax)-len(bx)))
    max_offset = min(len(ax), offset_bound + min(0, len(ax)-len(bx)))

    # Create a shorter series by summing small sequences of consecutive samples.
    # Since |a+b|<=|a|+|b|, the metric over sums is a lower bound on the metric
    # over the original series, so we can reject many offsets earlier.

    # The groups must line up between a and the delayed b, so the groups for b
    # are recalculated at different shifts, e.g. shift=1:
    #   a [-|-----|-----|-----|-...-|----]
    #   b   [-----|-----|-----|-...-|--]

    # combine channels for grouping since they are correlated anyway
    am = np.sum(ax, axis=1, dtype=ax.dtype)
    bm = np.sum(bx, axis=1, dtype=bx.dtype)

    # Group size chosen empirically. Best value probably depends on sample rate,
    # track frequences, and cache size. Check for possibility of overflow when
    # summing samples in baseline and optimized _limited_ds() before increasing.
    group = 59
    bg = _group_sums(bm, group)

    # First shift iteration covers offset=0, so if tracks are identical, limit
    # becomes 0 which speeds up the rest of the search.
    def iteration(shift):
        ag = _group_sums(am[shift:], group)

        # Range derived from this condition:
        #   min_offset <= i*group+shift <= max_offset
        start = -(-(min_offset-shift)//group)
        stop = (max_offset-shift)//group+1
        _cmp_candidates(ag, bg, start, stop, limit[0],
                        lambda i: check_match(i*group+shift))

    if _use_threads and threads != 0: # None means automatic
        with contextlib.closing(multiprocessing.dummy.Pool(threads)) as pool:
            list(pool.map(iteration, xrange(group), chunksize=1))
    else:
        list(map(iteration, xrange(group)))

    # sort for ordered offset for equal MAD, offset=0 is best
    matches = sorted(matches, key=lambda x: (x[0], abs(x[1]), x[1]<0))
    for ds, offset in matches:
        match = Match(a, b, offset)
        assert ds == match.ds()
        yield match


def cmp_album(a, b, skip=None, **options):
    if skip is None:
        skip = 0
    assert skip >= 0

    assert a.rate == b.rate
    if not skip and len(a.tracks) != len(b.tracks):
        return

    tracks = []
    at = list(a.tracks)
    bt = list(b.tracks)
    while True:
        for i, j in sorted(itertools.product(xrange(min(len(at), skip+1)),
                                             xrange(min(len(bt), skip+1))),
                           key=lambda x: (sum(x), abs(x[0]-x[1]), x[0]>x[1])):
            matches = list(cmp_track(at[i], bt[j], **options))
            if matches:
                tracks += [[Match(t, None, 0)] for t in at[:i]]
                tracks += [[Match(None, t, 0)] for t in bt[:j]]
                tracks.append(matches)
                del at[:i+1]
                del bt[:j+1]
                break
        else:
            if len(at) <= skip and len(bt) <= skip:
                tracks += [[Match(t, None, 0)] for t in at]
                tracks += [[Match(None, t, 0)] for t in bt]
                break
            else:
                return

    # The cartesian product of all track matches may be huge. Instead, we list
    # only some interesting combinations.

    # First tier album matches have matched padding between all tracks.
    # There is at most one album match possible for each first track match.
    # Album matches are ordered by overall difference metric.

    assert tracks # at least one track
    sequences = [[i] for i in tracks[0]]
    for track in tracks[1:]:
        ends = {i[-1].end_offset(): i for i in sequences}
        matches = {i.offset: i for i in track}
        sequences = [ends[i] + [matches[i]] for i in set(ends) & set(matches)]
    matches = [MatchSequence(a, b, i) for i in sequences]
    for match in sorted(matches,
                        key=lambda x: (x.ds(), abs(x.sequence[0].offset),
                                       x.sequence[0].offset<0)):
        yield match

    # Second tier album match is the single combination of best track matches,
    # unless it's already listed as a first tier match.

    best = [track[0] for track in tracks]
    for match in matches:
        if all(x is y for x, y in zip(best, match.sequence)):
            break
    else:
        yield MatchSequence(a, b, best)

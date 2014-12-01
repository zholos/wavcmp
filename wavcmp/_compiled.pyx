# These are optimized implementations of baseline routines from cmp.py.
# See there for general comments. Comments here are about optimizations.

cimport cython
cimport numpy as np

DEF step = 1<<15 # larger than in baseline because we can check limit mid-step
DEF window = 128

ctypedef unsigned long long uint64
cdef extern from "limits.h":
    cdef uint64 ULLONG_MAX

# Initial state for _limited_ds() only depends on n, and is valid for smaller n.
# Here we set up the state in advance and store it in this structure.
cdef uint64 c = 2305843009213693951
cdef struct _lds_info:
    size_t i0, n
    uint64 a

@cython.cdivision(True) # same as Python division because all positive
cdef _lds_init(_lds_info* _, size_t size):
    _.n = (max(size, 1)-1)//step+1
    # uint64 should be large enough, but ensure no overflow in i0= or i=.
    assert _.n <= (ULLONG_MAX-c) // 4 // _.n
    assert _.n % c
    _.a = _.n * 4 + 1
    _.i0 = c % _.n

@cython.cdivision(True)
cdef uint64 _lds_run(np.int32_t* ac, np.int32_t* bc, size_t size,
                     _lds_info* _, uint64 limit) nogil:

    # In baseline algorithm, sum of each chunk is done in int64 but chunk count
    # is unbounded. But uint64 should actually be sufficient for entire track.
    cdef uint64 s = 0

    cdef size_t i = _.i0
    cdef (np.int32_t*) ai, bi
    cdef np.int32_t d, t # 128 groups of 59 samples still fits in int32
    cdef unsigned m, w, j, k
    while True:
        ai = &ac[i*step]
        bi = &bc[i*step]
        m = min(step, size-i*step)

        # Sum over smaller fixed-size windows without checking limit,
        # to allow unrolling and vectorization.
        w = m - m % window
        for j in xrange(0, w, window):
            t = 0
            for k in xrange(window):
                d = ai[j+k] - bi[j+k]
                t += d if d>0 else -d # abs() is more complex
            s += t
            if s > limit:
                return s

        for j in xrange(w, m):
            d = ai[j] - bi[j]
            s += d if d>0 else -d

        if i == 0:
            return s
        i = (_.a * i + c) % _.n

@cython.boundscheck(False) # needed in case len(ac)==0
def limited_ds(np.ndarray[np.int32_t, ndim=2] ac,
               np.ndarray[np.int32_t, ndim=2] bc, uint64 limit):

    # Should only be called for ungrouped data with order="F". In this case each
    # channel will be contiguous and assigning to [::1] array will not cause an
    # error. However, since this is a subarray from the source, both channels
    # will not be contiguous together and must be processed separately.

    assert ac.shape[0] == bc.shape[0]
    assert ac.shape[1] == bc.shape[1]

    cdef _lds_info _
    _lds_init(&_, ac.shape[0])

    cdef np.int32_t[::1] ar, br
    cdef uint64 s, total = 0

    for channel in xrange(ac.shape[1]):
        ar = ac[:,channel]
        br = bc[:,channel]
        s = _lds_run(&ar[0], &br[0], ac.shape[0], &_, limit)
        if s > limit:
            return
        limit -= s
        total += s
    return total

@cython.boundscheck(False)
def cmp_candidates(np.int32_t[::1] ag, np.int32_t[::1] bg,
                   int start, int stop, uint64 limit, check_match):
    # Cython catches overflow in typed arguments on entry

    cdef _lds_info _
    _lds_init(&_, min(ag.shape[0], bg.shape[0]))

    cdef size_t size
    cdef uint64 s
    cdef int i, a0, b0

    with nogil:
        for i in xrange(start, stop):
            a0 = max(0, i)
            b0 = min(0, i) # avoid possible overflow in -i
            size = min(ag.shape[0]-a0, bg.shape[0]+b0)
            s = _lds_run(&ag[0]+a0, &bg[0]-b0, size, &_, limit)
            if s <= limit:
                with gil:
                    limit = check_match(i)

algorithm = "Cython"

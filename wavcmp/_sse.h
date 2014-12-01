#if defined __SSE2__ && defined __SSSE3__

#include <emmintrin.h> // SSE2
#include <tmmintrin.h> // SSSE3

// 64 is a good window size, but the loop is only automatically unrolled by
// Clang at 32, and not at all by GCC. Therefore, unroll manually.

#define _sse_window 64
#define REPEAT_16(e) e e e e e e e e e e e e e e e e

// C compiler ensures this isn't called with np.int32_t != int
static int
_wds_sse(int* a_, int* b_)
{
    __m128i* a = (__m128i*)a_;
    __m128i* b = (__m128i*)b_;
    __m128i t = _mm_setzero_si128(); // optimized out of first iteration
    REPEAT_16(
        t = _mm_add_epi32(t, _mm_abs_epi32(_mm_sub_epi32(
                _mm_loadu_si128(a++), _mm_loadu_si128(b++))));
    )
    return _mm_cvtsi128_si32(_mm_hadd_epi32(_mm_hadd_epi32(t, t), t));
}

#else

// dynamically unused, should be optimized out
#define _sse_window 0
static int _wds_sse(int* a, int* b) { return 0; }

#endif

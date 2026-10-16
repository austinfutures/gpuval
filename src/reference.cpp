#include "reference.hpp"
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#define GPUVAL_X86 1
#endif

namespace ref {

static _Float16 as_h(uint16_t b) { _Float16 h; std::memcpy(&h, &b, 2); return h; }
static uint16_t bits(_Float16 h) { uint16_t b; std::memcpy(&b, &h, 2); return b; }

#ifdef GPUVAL_X86
bool has_f16c() {
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    return (c & bit_F16C) != 0;
}
__attribute__((target("f16c"))) uint16_t hw_from_float(float f) {
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
}
__attribute__((target("f16c"))) float hw_to_float(uint16_t h) {
    return _cvtsh_ss(h);
}
#else
bool has_f16c() { return false; }
// No F16C: fall back to the compiler's conversion so the binary still runs.
uint16_t hw_from_float(float f) { return bits((_Float16)f); }
float hw_to_float(uint16_t h) { return (float)as_h(h); }
#endif

// double holds a + b and a * b of two halves exactly (<= 40 significant bits),
// so the only rounding is the final double -> _Float16 conversion.
uint16_t add(uint16_t a, uint16_t b) { return bits((_Float16)((double)as_h(a) + (double)as_h(b))); }
uint16_t mul(uint16_t a, uint16_t b) { return bits((_Float16)((double)as_h(a) * (double)as_h(b))); }

#ifdef __SIZEOF_FLOAT128__
bool has_fma_reference() { return true; }
// a*b + c can need ~80 bits; __float128 has 113, so it is still exact.
uint16_t fma(uint16_t a, uint16_t b, uint16_t c) {
    __float128 r = (__float128)as_h(a) * (__float128)as_h(b) + (__float128)as_h(c);
    return bits((_Float16)r);
}
#else
bool has_fma_reference() { return false; }
uint16_t fma(uint16_t, uint16_t, uint16_t) { return 0; }
#endif

}  // namespace ref

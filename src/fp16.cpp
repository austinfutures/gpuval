// fp16.cpp — integer-only implementation of the binary16 model.
//
// Design: every operation first computes the EXACT result as
//     value = (-1)^sign * sig * 2^exp        (sig is a wide integer)
// and then calls a single rounding routine, round_pack(). Keeping one rounding
// point mirrors how a real FPU datapath is built (align -> add/multiply ->
// normalize -> round) and means add, mul, fma and conversion all share the
// same, separately tested rounding logic.
#include "fp16.hpp"
#include <cstring>

namespace fp16 {

Faults faults;

using u128 = unsigned __int128;
using i128 = __int128;

static constexpr uint16_t QNAN = 0x7E00;  // canonical quiet NaN
static constexpr uint16_t INF  = 0x7C00;

static int msb(u128 x) {  // index of highest set bit, -1 if x == 0
    uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
    if (hi) return 64 + 63 - __builtin_clzll(hi);
    return lo ? 63 - __builtin_clzll(lo) : -1;
}

// Round (-1)^sign * sig * 2^exp to binary16, round-to-nearest-even.
static uint16_t round_pack(uint32_t sign, int exp, u128 sig) {
    const uint16_t s = (uint16_t)(sign << 15);
    if (sig == 0) return s;

    const int p = msb(sig);
    // Keep 11 significant bits for a normal result; but the smallest step the
    // format can represent is 2^-24 (the subnormal quantum), so never keep
    // bits finer than that.
    int shift = p - 10;
    if (shift < -24 - exp) shift = -24 - exp;

    uint64_t m;  // rounded significand, in units of 2^(exp + shift)
    if (shift <= 0) {
        m = (uint64_t)(sig << -shift);  // exact, no rounding needed
    } else if (shift > p + 1) {
        m = 0;                          // less than half the smallest step
    } else {
        const u128 kept = sig >> shift;
        const u128 rem  = sig - (kept << shift);
        const u128 half = (u128)1 << (shift - 1);
        m = (uint64_t)kept;
        if (faults.round_half_up) {
            if (rem >= half) m++;                       // BUG: ties away
        } else if (rem > half || (rem == half && (m & 1))) {
            m++;                                        // ties to even
        }
    }

    int q = exp + shift;              // exponent of m's least significant bit
    if (m == 2048) { m = 1024; q++; }  // rounding carried into a new bit

    if (m >= 1024) {                  // normal number
        const int biased = q + 10 + 15;
        if (biased >= 31) return s | INF;  // overflow
        return (uint16_t)(s | (biased << 10) | (m & 0x3FF));
    }
    // subnormal (q == -24 here). Exponent field is 0.
    if (faults.flush_subnormals) return s;  // BUG: flush to zero
    return (uint16_t)(s | m);
}

// Split a finite half into significand and exponent: value = sig * 2^exp.
static void unpack(uint16_t h, uint32_t& sign, int& exp, uint32_t& sig) {
    sign = h >> 15;
    const int e = (h >> 10) & 0x1F;
    const uint32_t m = h & 0x3FF;
    if (e == 0) { sig = m;        exp = -24;    }  // subnormal or zero
    else        { sig = m | 1024; exp = e - 25; }  // implicit leading 1
}

uint16_t from_float(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t sign = b >> 31;
    const int e = (b >> 23) & 0xFF;
    const uint32_t m = b & 0x7FFFFF;
    const uint16_t s = (uint16_t)(sign << 15);
    if (e == 255) {
        if (m) return (uint16_t)(s | 0x7C00 | 0x200 | (m >> 13));  // quiet, keep payload
        return s | INF;
    }
    if (e == 0) return round_pack(sign, -149, m);           // float subnormal
    return round_pack(sign, e - 150, m | 0x800000);
}

float to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    int e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t bits;
    if (e == 31) {
        // Inf stays inf; a NaN keeps its payload and is quieted (bit 22 set),
        // matching what the hardware conversion does with signaling NaNs.
        bits = sign | 0x7F800000 | (m << 13) | (m ? 0x400000u : 0u);
    } else if (e == 0) {
        if (m == 0) {
            bits = sign;
        } else {  // normalize the subnormal: every half is a normal float
            e = 1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            bits = sign | (uint32_t)(e - 15 + 127) << 23 | ((m & 0x3FF) << 13);
        }
    } else {
        bits = sign | (uint32_t)(e - 15 + 127) << 23 | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t add(uint16_t a, uint16_t b) {
    if (is_nan(a) || is_nan(b)) return QNAN;
    if (is_inf(a) || is_inf(b)) {
        if (is_inf(a) && is_inf(b) && ((a ^ b) & 0x8000)) return QNAN;  // inf - inf
        return is_inf(a) ? a : b;
    }
    uint32_t sa, sb, ma, mb;
    int ea, eb;
    unpack(a, sa, ea, ma);
    unpack(b, sb, eb, mb);
    // Align to the smaller exponent. Max distance is 29 bits -> exact in 64.
    const int e = ea < eb ? ea : eb;
    int64_t va = (int64_t)ma << (ea - e);
    int64_t vb = (int64_t)mb << (eb - e);
    if (sa) va = -va;
    if (sb) vb = -vb;
    const int64_t sum = va + vb;
    if (sum == 0) return (uint16_t)((sa & sb) << 15);  // -0 only if -0 + -0
    return round_pack(sum < 0, e, (u128)(sum < 0 ? -sum : sum));
}

uint16_t mul(uint16_t a, uint16_t b) {
    const uint32_t sign = (a ^ b) >> 15;
    if (is_nan(a) || is_nan(b)) return QNAN;
    const bool za = (a & 0x7FFF) == 0, zb = (b & 0x7FFF) == 0;
    if (is_inf(a) || is_inf(b)) {
        if (za || zb) return QNAN;                    // inf * 0
        return (uint16_t)((sign << 15) | INF);
    }
    uint32_t sa, sb, ma, mb;
    int ea, eb;
    unpack(a, sa, ea, ma);
    unpack(b, sb, eb, mb);
    return round_pack(sign, ea + eb, (u128)ma * mb);  // 22-bit product, exact
}

uint16_t fma(uint16_t a, uint16_t b, uint16_t c) {
    if (faults.fma_unfused) return add(mul(a, b), c);  // BUG: two roundings

    if (is_nan(a) || is_nan(b) || is_nan(c)) return QNAN;
    const uint32_t sp = (a ^ b) >> 15;
    const bool za = (a & 0x7FFF) == 0, zb = (b & 0x7FFF) == 0;
    if (is_inf(a) || is_inf(b)) {
        if (za || zb) return QNAN;                              // inf * 0
        if (is_inf(c) && ((c >> 15) != sp)) return QNAN;        // inf - inf
        return (uint16_t)((sp << 15) | INF);
    }
    if (is_inf(c)) return c;

    uint32_t sa, sb, sc, ma, mb, mc;
    int ea, eb, ec;
    unpack(a, sa, ea, ma);
    unpack(b, sb, eb, mb);
    unpack(c, sc, ec, mc);
    // Exact product and exact sum. Alignment can need ~80 bits, so use 128.
    const int ep = ea + eb;
    const int e = ep < ec ? ep : ec;
    i128 vp = (i128)((u128)ma * mb) << (ep - e);
    i128 vc = (i128)mc << (ec - e);
    if (sp) vp = -vp;
    if (sc) vc = -vc;
    const i128 sum = vp + vc;
    if (sum == 0) {
        const bool prod_zero = (ma == 0 || mb == 0);
        // Exact zero: -0 only when both terms are -0, else +0 (RNE rule).
        return (uint16_t)(((prod_zero && mc == 0) ? (sp & sc) : 0) << 15);
    }
    return round_pack(sum < 0, e, (u128)(sum < 0 ? -sum : sum));
}

}  // namespace fp16

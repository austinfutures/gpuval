// fp16.hpp — bit-accurate model of an IEEE 754 binary16 (half precision) unit.
//
// Everything here is integer-only: no host floating-point arithmetic is used to
// produce a result. That is the point of a "bit-accurate C model" — it describes
// exactly which bits the hardware must produce, including rounding, subnormals,
// signed zeros, infinities and NaN handling, so it can serve as the golden
// reference for RTL simulation and post-silicon checking.
//
// Format: [15] sign | [14:10] exponent (bias 15) | [9:0] mantissa
// Rounding: round-to-nearest, ties-to-even (IEEE default).
#pragma once
#include <cstdint>

namespace fp16 {

// Fault-injection knobs. Each one plants a realistic hardware bug in the model
// so the test suites can prove they would catch it. All off by default.
struct Faults {
    bool round_half_up    = false;  // ties round away from zero instead of to even
    bool flush_subnormals = false;  // FTZ: tiny results become 0 (non-IEEE)
    bool fma_unfused      = false;  // FMA rounds the product before the add
};
extern Faults faults;

uint16_t from_float(float f);   // float32 -> binary16, RNE, NaN payload preserved
float    to_float(uint16_t h);  // binary16 -> float32, always exact

uint16_t add(uint16_t a, uint16_t b);
uint16_t mul(uint16_t a, uint16_t b);
uint16_t fma(uint16_t a, uint16_t b, uint16_t c);  // a*b + c with ONE rounding

inline bool is_nan(uint16_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x03FF); }
inline bool is_inf(uint16_t h) { return (h & 0x7FFF) == 0x7C00; }

// Two results match if the bits are identical, or both are NaN (NaN payload
// propagation is implementation-defined, so we only require "is a NaN").
inline bool same(uint16_t a, uint16_t b) { return a == b || (is_nan(a) && is_nan(b)); }

}  // namespace fp16

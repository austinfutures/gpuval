// reference.hpp — independent "golden" implementations the model is checked against.
//
// Two independent sources of truth, neither written by us:
//   1. Silicon: x86 F16C instructions (VCVTPS2PH / VCVTPH2PS) for conversions.
//   2. Compiler runtime: GCC/Clang's _Float16 soft-float library for arithmetic.
//      The exact result is formed in a wider type (double / __float128, which is
//      exact for these operations) and rounded to half ONCE by the runtime.
#pragma once
#include <cstdint>

namespace ref {

bool has_f16c();                     // CPU supports F16C conversion instructions
uint16_t hw_from_float(float f);     // VCVTPS2PH, round-to-nearest-even
float    hw_to_float(uint16_t h);    // VCVTPH2PS

bool has_fma_reference();            // needs __float128 (x86-64 GCC/Clang)
uint16_t add(uint16_t a, uint16_t b);
uint16_t mul(uint16_t a, uint16_t b);
uint16_t fma(uint16_t a, uint16_t b, uint16_t c);

}  // namespace ref

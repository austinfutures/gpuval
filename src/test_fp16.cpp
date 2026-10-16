// test_fp16.cpp — verification of the binary16 model.
//
// Strategy (the same ladder used for real FPU sign-off):
//   1. Exhaustive where the input space is small enough: every half (2^16),
//      every float32 (2^32), every pair of halves for add and mul (2^32 each).
//   2. Directed corner cases: zeros, subnormals, min/max normals, ties, inf, NaN,
//      crossed with each other.
//   3. Constrained-random where exhaustive is impossible (FMA has 2^48 inputs).
// Every check compares bits against an independent reference (see reference.hpp).
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>

#include "fp16.hpp"
#include "reference.hpp"
#include "util.hpp"

// Corner-case operands: signed zeros, smallest/largest subnormals, smallest
// normal, values around 1.0 (ties live here), largest finite, infinities, NaNs.
static const uint16_t kEdge[] = {
    0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x0003, 0x01FF, 0x0200, 0x03FF, 0x83FF,
    0x0400, 0x8400, 0x0401, 0x07FF, 0x0800, 0x1400, 0x2C00, 0x3555, 0x3800, 0x3BFF,
    0x3C00, 0xBC00, 0x3C01, 0xBC01, 0x3C02, 0x3E00, 0x4000, 0xC000, 0x4001, 0x57FF,
    0x5800, 0x6400, 0x67FF, 0x6800, 0x7800, 0x7BFE, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
    0x7E00, 0x7C01, 0xFE00, 0x7FFF,
};
static constexpr int kNumEdge = sizeof(kEdge) / sizeof(kEdge[0]);

// Run f(i) for i in [0, n) across threads. f returns false + message on mismatch.
// Keeps the failure with the LOWEST index so results are deterministic.
template <class F>
static void parallel_check(Check& c, uint64_t n, unsigned threads, F f) {
    Timer tm;
    std::mutex mu;
    std::atomic<uint64_t> fails{0};
    uint64_t first_idx = UINT64_MAX;
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; t++) {
        pool.emplace_back([&, t] {
            const uint64_t lo = n * t / threads, hi = n * (t + 1) / threads;
            uint64_t local = 0;
            std::string msg;
            for (uint64_t i = lo; i < hi; i++) {
                if (!f(i, msg)) {
                    if (local++ == 0) {
                        std::lock_guard<std::mutex> g(mu);
                        if (i < first_idx) { first_idx = i; c.first_failure = msg; }
                    }
                }
            }
            fails += local;
        });
    }
    for (auto& th : pool) th.join();
    c.vectors += n;
    c.failures += fails;
    c.seconds += tm.sec();
}

static std::string binop_msg(const char* op, uint16_t a, uint16_t b, uint16_t got, uint16_t want) {
    return std::string(op) + "(" + hex16(a) + ", " + hex16(b) + ") = " + hex16(got) + ", expected " + hex16(want);
}

std::vector<Check> run_fp16_suite(const Options& o) {
    const unsigned T = o.threads ? o.threads : std::max(1u, std::thread::hardware_concurrency());
    std::vector<Check> out;
    auto finish = [&](Check& c) { if (o.verbose) print_check(c); out.push_back(c); };

    if (o.verbose)
        std::printf("\n== fp16: bit-accurate half-precision model ==  (%u threads, reference: %s)\n", T,
                    ref::has_f16c() ? "x86 F16C silicon + compiler soft-float" : "compiler soft-float only");

    {   // 1. half -> float, every one of the 65,536 encodings
        Check c{"to_float  exhaustive"};
        parallel_check(c, 1u << 16, T, [](uint64_t i, std::string& m) {
            const uint16_t h = (uint16_t)i;
            float a = fp16::to_float(h), b = ref::hw_to_float(h);
            if (std::memcmp(&a, &b, 4) == 0) return true;
            m = "to_float(" + hex16(h) + ") bits differ from hardware";
            return false;
        });
        finish(c);
    }
    {   // 2. float -> half: all 4.29 billion float32 bit patterns
        Check c{o.quick ? "from_float random" : "from_float exhaustive"};
        const uint64_t n = o.quick ? (1ull << 24) : (1ull << 32);
        parallel_check(c, n, T, [&](uint64_t i, std::string& m) {
            const uint32_t b = o.quick ? (uint32_t)Rng(i).next() : (uint32_t)i;
            float f;
            std::memcpy(&f, &b, 4);
            const uint16_t got = fp16::from_float(f), want = ref::hw_from_float(f);
            if (got == want) return true;  // bit-exact, NaN payload included
            char buf[96];
            std::snprintf(buf, sizeof buf, "from_float(0x%08X) = %s, expected %s", b, hex16(got).c_str(), hex16(want).c_str());
            m = buf;
            return false;
        });
        finish(c);
    }
    {   // 3. round trip property: every non-NaN half survives half->float->half
        Check c{"round_trip exhaustive"};
        parallel_check(c, 1u << 16, T, [](uint64_t i, std::string& m) {
            const uint16_t h = (uint16_t)i;
            if (fp16::is_nan(h) || fp16::from_float(fp16::to_float(h)) == h) return true;
            m = "round trip changed " + hex16(h);
            return false;
        });
        finish(c);
    }

    // 4. add / mul: directed corner cases, then all 2^32 operand pairs
    struct Op { const char* name; uint16_t (*dut)(uint16_t, uint16_t); uint16_t (*gold)(uint16_t, uint16_t); };
    const Op ops[] = {{"add", fp16::add, ref::add}, {"mul", fp16::mul, ref::mul}};
    for (const Op& op : ops) {
        Check d{std::string(op.name) + "       directed"};
        parallel_check(d, kNumEdge * kNumEdge, 1, [&](uint64_t i, std::string& m) {
            const uint16_t a = kEdge[i / kNumEdge], b = kEdge[i % kNumEdge];
            const uint16_t got = op.dut(a, b), want = op.gold(a, b);
            if (fp16::same(got, want)) return true;
            m = binop_msg(op.name, a, b, got, want);
            return false;
        });
        finish(d);

        Check c{std::string(op.name) + (o.quick ? "       random" : "       exhaustive")};
        const uint64_t n = o.quick ? (1ull << 22) : (1ull << 32);
        parallel_check(c, n, T, [&](uint64_t i, std::string& m) {
            const uint32_t ab = o.quick ? (uint32_t)Rng(i ^ 0xADD).next() : (uint32_t)i;
            const uint16_t a = (uint16_t)(ab >> 16), b = (uint16_t)ab;
            const uint16_t got = op.dut(a, b), want = op.gold(a, b);
            if (fp16::same(got, want)) return true;
            m = binop_msg(op.name, a, b, got, want);
            return false;
        });
        finish(c);
    }

    // 5. FMA: 2^48 inputs is too many, so directed cube + constrained random.
    {
        Check d{"fma       directed"};
        Check r{"fma       random"};
        if (!ref::has_fma_reference()) {
            d.skipped = r.skipped = true;
        } else {
            auto check3 = [](uint16_t a, uint16_t b, uint16_t c, std::string& m) {
                const uint16_t got = fp16::fma(a, b, c), want = ref::fma(a, b, c);
                if (fp16::same(got, want)) return true;
                m = "fma(" + hex16(a) + ", " + hex16(b) + ", " + hex16(c) + ") = " + hex16(got) + ", expected " + hex16(want);
                return false;
            };
            parallel_check(d, (uint64_t)kNumEdge * kNumEdge * kNumEdge, T, [&](uint64_t i, std::string& m) {
                return check3(kEdge[i % kNumEdge], kEdge[(i / kNumEdge) % kNumEdge], kEdge[i / kNumEdge / kNumEdge], m);
            });
            // Constrained random: half the time pick c near -a*b so the add
            // cancels massively — the case where fused vs unfused differ most.
            const uint64_t n = o.quick ? (1ull << 21) : (1ull << 26);
            parallel_check(r, n, T, [&](uint64_t i, std::string& m) {
                Rng g(i * 7919 + 13);
                const uint16_t a = (uint16_t)g.next(), b = (uint16_t)g.next();
                uint16_t c = (uint16_t)g.next();
                if (g.next() & 1) c = (uint16_t)((fp16::mul(a, b) ^ 0x8000) + (int)(g.next() % 5) - 2);
                return check3(a, b, c, m);
            });
        }
        finish(d);
        finish(r);
    }
    return out;
}

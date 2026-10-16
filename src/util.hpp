// util.hpp — small helpers: deterministic RNG, CRC32, BMP writer, test bookkeeping.
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// SplitMix64: tiny, fast, and identical on every platform/compiler, which keeps
// every test vector and golden signature reproducible (std:: distributions are not).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int64_t range(int64_t lo, int64_t hi) { return lo + (int64_t)(next() % (uint64_t)(hi - lo + 1)); }
};

inline uint32_t crc32(const void* data, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

// 24-bit BMP. VS Code can preview .bmp files directly.
inline bool write_bmp(const char* path, int w, int h, const std::vector<uint32_t>& argb) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const int stride = (w * 3 + 3) & ~3;
    const uint32_t size = 54 + stride * h;
    uint8_t hdr[54] = {'B', 'M'};
    auto put32 = [&](int off, uint32_t v) { for (int i = 0; i < 4; i++) hdr[off + i] = (uint8_t)(v >> (8 * i)); };
    put32(2, size); put32(10, 54); put32(14, 40); put32(18, w); put32(22, h);
    hdr[26] = 1; hdr[28] = 24; put32(34, stride * h);
    std::fwrite(hdr, 1, 54, f);
    std::vector<uint8_t> row(stride, 0);
    for (int y = h - 1; y >= 0; y--) {  // BMP rows are stored bottom-up
        for (int x = 0; x < w; x++) {
            const uint32_t c = argb[y * w + x];
            row[x * 3 + 0] = c & 0xFF; row[x * 3 + 1] = (c >> 8) & 0xFF; row[x * 3 + 2] = (c >> 16) & 0xFF;
        }
        std::fwrite(row.data(), 1, stride, f);
    }
    std::fclose(f);
    return true;
}

// One named check: how many vectors ran, how many failed, and the first failure.
struct Check {
    Check(std::string n = "") : name(std::move(n)) {}
    std::string name;
    uint64_t vectors = 0, failures = 0;
    std::string first_failure;
    double seconds = 0;
    bool skipped = false;
    bool passed() const { return skipped || failures == 0; }
    void fail(const std::string& msg) { if (failures++ == 0) first_failure = msg; }
};

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double sec() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

inline std::string hex16(uint16_t v) { char b[8]; std::snprintf(b, sizeof b, "0x%04X", v); return b; }

inline std::string commas(uint64_t v) {
    std::string s = std::to_string(v);
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert(i, ",");
    return s;
}

inline void print_check(const Check& c) {
    const char* tag = c.skipped ? "SKIP" : (c.failures ? "FAIL" : "PASS");
    std::printf("  [%s] %-32s %16s vectors  %8s fails  %7.2f s\n", tag, c.name.c_str(),
                commas(c.vectors).c_str(), commas(c.failures).c_str(), c.seconds);
    if (c.failures) std::printf("         first failure: %s\n", c.first_failure.c_str());
}

// Test suites (defined in test_fp16.cpp / test_raster.cpp)
struct Options { bool quick = false; bool verbose = true; unsigned threads = 0; };
std::vector<Check> run_fp16_suite(const Options& o);
std::vector<Check> run_raster_suite(const Options& o);

// test_raster.cpp — verification of the tiled rasterizer (DUT).
//
//   watertight   Meshes that tile the whole screen with shared edges. Every
//                pixel must be covered EXACTLY once: 0 = crack, 2 = overdraw.
//                Checks the fill rule without needing any reference image.
//   differential Random triangle soup drawn by DUT and golden model; color,
//                depth and coverage must match bit-for-bit after every triangle.
//   signature    CRC32 of a fixed scene must equal a stored golden value, the
//                way post-silicon tests check a frame without shipping images.
#include <algorithm>
#include <cstring>

#include "raster.hpp"
#include "util.hpp"

using raster::Vertex;
using raster::ONE;
using raster::HALF;

static Vertex vtx(int64_t x, int64_t y, Rng& g) {
    return Vertex{(int32_t)x, (int32_t)y, (uint32_t)(g.next() & 0xFFFFFF),
                  (uint8_t)g.next(), (uint8_t)g.next(), (uint8_t)g.next()};
}

// Shuffle vertex order so both windings and every starting vertex get exercised.
static void add_tri(std::vector<Vertex>& tris, Vertex a, Vertex b, Vertex c, Rng& g) {
    Vertex v[3] = {a, b, c};
    for (int i = 2; i > 0; i--) std::swap(v[i], v[g.next() % (i + 1)]);
    tris.insert(tris.end(), v, v + 3);
}

// Grid mesh covering [0,W]x[0,H]. Interior vertices are jittered or snapped to
// pixel centers (snapping puts edges exactly through pixel centers, i.e. ties).
static std::vector<Vertex> grid_mesh(int W, int H, int nx, int ny, bool snap_centers, Rng& g) {
    std::vector<Vertex> pts((nx + 1) * (ny + 1));
    for (int j = 0; j <= ny; j++) {
        for (int i = 0; i <= nx; i++) {
            int64_t x = (int64_t)i * W * ONE / nx, y = (int64_t)j * H * ONE / ny;
            const bool interior = i > 0 && i < nx && j > 0 && j < ny;
            if (interior && snap_centers) {
                x = (x / ONE) * ONE + HALF;
                y = (y / ONE) * ONE + HALF;
            } else if (interior) {
                const int64_t jx = (int64_t)W * ONE / nx / 5, jy = (int64_t)H * ONE / ny / 5;
                x += g.range(-jx, jx);
                y += g.range(-jy, jy);
            }
            pts[j * (nx + 1) + i] = vtx(x, y, g);
        }
    }
    std::vector<Vertex> tris;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            const Vertex& a = pts[j * (nx + 1) + i];
            const Vertex& b = pts[j * (nx + 1) + i + 1];
            const Vertex& c = pts[(j + 1) * (nx + 1) + i];
            const Vertex& d = pts[(j + 1) * (nx + 1) + i + 1];
            if (g.next() & 1) { add_tri(tris, a, b, d, g); add_tri(tris, a, d, c, g); }
            else              { add_tri(tris, a, b, c, g); add_tri(tris, b, d, c, g); }
        }
    }
    return tris;
}

// Fan from one center vertex to points walking around the screen border.
// Stresses a single very-high-valence vertex.
static std::vector<Vertex> fan_mesh(int W, int H, int n, Rng& g) {
    const int64_t w = (int64_t)W * ONE, h = (int64_t)H * ONE, per = 2 * (w + h);
    std::vector<int64_t> t;
    for (int i = 0; i < n; i++) t.push_back(g.range(0, per - 1));
    t.push_back(0); t.push_back(w); t.push_back(w + h); t.push_back(2 * w + h);  // corners
    std::sort(t.begin(), t.end());
    t.erase(std::unique(t.begin(), t.end()), t.end());
    auto border = [&](int64_t s) -> std::pair<int64_t, int64_t> {
        if (s < w) return {s, 0};
        if (s < w + h) return {w, s - w};
        if (s < 2 * w + h) return {w - (s - w - h), h};
        return {0, h - (s - 2 * w - h)};
    };
    const Vertex center = vtx((W / 2) * ONE + HALF, (H / 3) * ONE + HALF, g);  // on a pixel center
    std::vector<Vertex> tris;
    for (size_t i = 0; i < t.size(); i++) {
        auto [x0, y0] = border(t[i]);
        auto [x1, y1] = border(t[(i + 1) % t.size()]);
        add_tri(tris, center, vtx(x0, y0, g), vtx(x1, y1, g), g);
    }
    return tris;
}

static void check_watertight(Check& c, const char* what, int W, int H, const std::vector<Vertex>& tris) {
    raster::Target t(W, H);
    t.depth_test = false;
    for (size_t i = 0; i < tris.size(); i += 3) raster::draw_tiled(t, &tris[i]);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            c.vectors++;
            const int n = t.hits[y * W + x];
            if (n != 1) {
                char b[160];
                std::snprintf(b, sizeof b, "%s %dx%d: pixel (%d,%d) covered %d times (%s)", what, W, H, x, y, n,
                              n == 0 ? "crack" : "overdraw");
                c.fail(b);
            }
        }
    }
}

// Random soup: mix of slivers, sub-pixel tris, huge tris, off-screen and
// degenerate ones, and vertices snapped to pixel centers to force ties.
static std::vector<Vertex> soup(int W, int H, int n, Rng& g) {
    std::vector<Vertex> tris;
    for (int i = 0; i < n; i++) {
        const int kind = (int)(g.next() % 4);
        const int64_t r = kind == 0 ? ONE / 2 : kind == 1 ? 4 * ONE : kind == 2 ? 40 * ONE : 4 * W * ONE;
        const int64_t cx = g.range(-W / 4 * ONE, W * 5 / 4 * ONE), cy = g.range(-H / 4 * ONE, H * 5 / 4 * ONE);
        Vertex v[3];
        for (auto& p : v) {
            int64_t x = cx + g.range(-r, r), y = cy + g.range(-r, r);
            if (g.next() % 4 == 0) { x = (x >> raster::SUB << raster::SUB) + HALF; y = (y >> raster::SUB << raster::SUB) + HALF; }
            p = vtx(x, y, g);
        }
        if (g.next() % 50 == 0) v[2] = v[1];  // degenerate
        tris.insert(tris.end(), v, v + 3);
    }
    return tris;
}

// Fixed scene for the golden signature. Integer-only, so the CRC is identical
// on every machine and compiler.
static uint32_t scene_signature(bool dut, raster::Stats* st) {
    Rng g(2026);
    raster::Target t(320, 240);
    const auto tris = soup(320, 240, 3000, g);
    for (size_t i = 0; i < tris.size(); i += 3)
        dut ? raster::draw_tiled(t, &tris[i], st) : raster::draw_reference(t, &tris[i], st);
    uint32_t crc = crc32(t.color.data(), t.color.size() * 4);
    return crc32(t.depth.data(), t.depth.size() * 4, crc);
}
static constexpr uint32_t kGoldenSignature = 0x7A5D0FA6;  // produced by draw_reference()

std::vector<Check> run_raster_suite(const Options& o) {
    std::vector<Check> out;
    auto finish = [&](Check& c) { if (o.verbose) print_check(c); out.push_back(c); };
    if (o.verbose) std::printf("\n== raster: fixed-point tiled rasterizer ==\n");

    {   // Watertightness over several mesh styles and odd (non-tile-multiple) sizes
        Check c{"watertight meshes"};
        Timer tm;
        const int seeds = o.quick ? 3 : 20;
        for (int s = 0; s < seeds; s++) {
            Rng g(100 + s);
            const int W = 64 + (int)(g.next() % 200), H = 48 + (int)(g.next() % 150);
            check_watertight(c, "jittered grid", W, H, grid_mesh(W, H, 7 + s, 5 + s, false, g));
            check_watertight(c, "pixel-center grid", W, H, grid_mesh(W, H, 9 + s, 6 + s, true, g));
            check_watertight(c, "sub-pixel grid", W, H, grid_mesh(W, H, W * 2, H * 2, false, g));
            check_watertight(c, "fan", W, H, fan_mesh(W, H, 40 + 10 * s, g));
        }
        c.seconds = tm.sec();
        finish(c);
    }

    {   // Differential: DUT vs golden after every triangle, inside its bounds
        Check c{"differential vs golden"};
        Timer tm;
        const int seeds = o.quick ? 4 : 40;
        for (int s = 0; s < seeds && c.failures == 0; s++) {
            Rng g(7000 + s);
            const int W = 32 + (int)(g.next() % 300), H = 32 + (int)(g.next() % 200);
            const auto tris = soup(W, H, 1500, g);
            raster::Target dut(W, H), gold(W, H);
            for (size_t i = 0; i < tris.size() && c.failures == 0; i += 3) {
                raster::draw_tiled(dut, &tris[i]);
                raster::draw_reference(gold, &tris[i]);
                int x0, y0, x1, y1;
                if (!raster::pixel_bounds(dut, &tris[i], x0, y0, x1, y1)) { c.vectors++; continue; }
                for (int y = y0; y <= y1; y++) {
                    for (int x = x0; x <= x1; x++) {
                        const int p = y * W + x;
                        c.vectors++;
                        if (dut.color[p] != gold.color[p] || dut.depth[p] != gold.depth[p] || dut.hits[p] != gold.hits[p]) {
                            char b[200];
                            std::snprintf(b, sizeof b,
                                          "seed %d tri #%zu pixel (%d,%d): dut color %08X hits %d, golden color %08X hits %d",
                                          s, i / 3, x, y, dut.color[p], dut.hits[p], gold.color[p], gold.hits[p]);
                            c.fail(b);
                        }
                    }
                }
            }
        }
        c.seconds = tm.sec();
        finish(c);
    }

    {   // Golden frame signature + tiling efficiency stats
        Check c{"frame signature"};
        Timer tm;
        raster::Stats sd, sr;
        const uint32_t gold = scene_signature(false, &sr);
        const uint32_t dut = scene_signature(true, &sd);
        c.vectors = 2;
        char b[160];
        if (gold != kGoldenSignature) {
            std::snprintf(b, sizeof b, "golden model CRC %08X != stored %08X", gold, kGoldenSignature);
            c.fail(b);
        }
        if (dut != kGoldenSignature) {
            std::snprintf(b, sizeof b, "DUT frame CRC %08X != golden %08X", dut, kGoldenSignature);
            c.fail(b);
        }
        c.seconds = tm.sec();
        finish(c);
        if (o.verbose) {
            const uint64_t tiles = sd.tiles_rejected + sd.tiles_accepted + sd.tiles_partial;
            std::printf("         tiles: %s total, %.1f%% trivially rejected, %.1f%% trivially accepted, %.1f%% partial\n",
                        commas(tiles).c_str(), 100.0 * sd.tiles_rejected / tiles,
                        100.0 * sd.tiles_accepted / tiles, 100.0 * sd.tiles_partial / tiles);
            std::printf("         per-pixel edge tests: golden %s, tiled %s (%.1f%% fewer)\n",
                        commas(sr.pixel_tests).c_str(), commas(sd.pixel_tests).c_str(),
                        100.0 * (1.0 - (double)sd.pixel_tests / sr.pixel_tests));
        }
    }
    return out;
}

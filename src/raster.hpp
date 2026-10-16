// raster.hpp — fixed-point triangle rasterizer, modeled the way GPU hardware does it.
//
// Hardware rasterizers do not use floating point for coverage. Vertex positions
// are snapped to a fixed-point grid (here 24.8: 8 bits = 1/256 pixel of
// sub-pixel precision) and coverage is decided by exact integer edge functions.
// That makes the result bit-exact and reproducible, which is what lets us check
// it against a golden model and against invariants like "watertightness".
//
// Two implementations of the same spec:
//   draw_reference()  brute force, one full edge evaluation per pixel. Obviously
//                     correct, slow. This is the golden model.
//   draw_tiled()      8x8 tile traversal with trivial reject/accept and
//                     incremental edge stepping, like a hardware coarse+fine
//                     raster. This is the "device under test" (DUT).
#pragma once
#include <cstdint>
#include <vector>

namespace raster {

constexpr int SUB  = 8;              // sub-pixel bits
constexpr int ONE  = 1 << SUB;       // one pixel in fixed point
constexpr int HALF = ONE / 2;        // pixel centers sit at +0.5
constexpr int TILE = 8;              // tile edge length in pixels

struct Vertex {
    int32_t  x, y;      // screen position, 24.8 fixed point, y points down
    uint32_t z;         // depth, 24-bit unsigned, smaller = closer
    uint8_t  r, g, b;   // color
};

// Bugs that can be planted in the DUT (draw_tiled) only. Golden stays clean.
struct Faults {
    bool no_top_left       = false;  // shared-edge pixels drawn by both triangles
    bool tile_accept_3pt   = false;  // trivial-accept checks 3 of 4 tile corners
    bool interp_no_round   = false;  // attribute interpolation truncates
};
extern Faults faults;

struct Stats {
    uint64_t triangles = 0, degenerate = 0;
    uint64_t tiles_rejected = 0, tiles_accepted = 0, tiles_partial = 0;
    uint64_t pixel_tests = 0;     // per-pixel inside/outside evaluations
    uint64_t pixels_covered = 0;
};

struct Target {
    int w, h;
    bool depth_test = true;
    std::vector<uint32_t> color;  // 0xAARRGGBB
    std::vector<uint32_t> depth;
    std::vector<uint16_t> hits;   // coverage count, ignores depth test
    Target(int w, int h);
    void clear(uint32_t rgb = 0xFF000000);
};

void draw_reference(Target& t, const Vertex tri[3], Stats* st = nullptr);
void draw_tiled(Target& t, const Vertex tri[3], Stats* st = nullptr);

// Inclusive pixel bounding box a triangle can touch on this target.
// Returns false if it touches nothing.
bool pixel_bounds(const Target& t, const Vertex tri[3], int& x0, int& y0, int& x1, int& y1);

}  // namespace raster

#include "raster.hpp"
#include <algorithm>

namespace raster {

Faults faults;

Target::Target(int w_, int h_) : w(w_), h(h_), color(w_ * h_), depth(w_ * h_), hits(w_ * h_) { clear(); }

void Target::clear(uint32_t rgb) {
    std::fill(color.begin(), color.end(), rgb);
    std::fill(depth.begin(), depth.end(), 0xFFFFFFu);
    std::fill(hits.begin(), hits.end(), 0);
}

// Edge function for directed edge a->b at point p (all fixed point).
// Equals twice the signed area of triangle (a, b, p). Exact in 64-bit.
static inline int64_t edge(const Vertex& a, const Vertex& b, int64_t px, int64_t py) {
    return (int64_t)(b.x - a.x) * (py - a.y) - (int64_t)(b.y - a.y) * (px - a.x);
}

// Top-left fill rule (same as D3D/Vulkan/Metal): a pixel center lying exactly on
// an edge belongs to the triangle only if that edge is a "top" or "left" edge.
// This is what makes adjacent triangles share an edge with no gaps and no
// double-drawn pixels. With our orientation (area > 0, y down) a top edge is
// horizontal going right and a left edge goes up.
static inline bool is_top_left(const Vertex& a, const Vertex& b) {
    const int dx = b.x - a.x, dy = b.y - a.y;
    return dy < 0 || (dy == 0 && dx > 0);
}

static inline int64_t floor_div(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
static inline int64_t ceil_div(int64_t a, int64_t b) { return -floor_div(-a, b); }

struct Setup {
    Vertex v[3];
    int64_t area;
    int64_t bias[3];   // 0 or -1: "E + bias >= 0" means inside
    int x0, y0, x1, y1;
};

// Shared triangle setup. Returns false for degenerate or fully off-screen tris.
static bool setup(const Target& t, const Vertex in[3], bool dut, Setup& s) {
    s.v[0] = in[0]; s.v[1] = in[1]; s.v[2] = in[2];
    s.area = edge(s.v[0], s.v[1], s.v[2].x, s.v[2].y);
    if (s.area == 0) return false;
    if (s.area < 0) { std::swap(s.v[1], s.v[2]); s.area = -s.area; }  // cull none
    // Edge i is the one opposite vertex i; its edge function is vertex i's weight.
    const Vertex* e[3][2] = {{&s.v[1], &s.v[2]}, {&s.v[2], &s.v[0]}, {&s.v[0], &s.v[1]}};
    for (int i = 0; i < 3; i++) {
        const bool tl = is_top_left(*e[i][0], *e[i][1]);
        s.bias[i] = (dut && faults.no_top_left) ? 0 : (tl ? 0 : -1);
    }
    const int64_t minx = std::min({s.v[0].x, s.v[1].x, s.v[2].x});
    const int64_t maxx = std::max({s.v[0].x, s.v[1].x, s.v[2].x});
    const int64_t miny = std::min({s.v[0].y, s.v[1].y, s.v[2].y});
    const int64_t maxy = std::max({s.v[0].y, s.v[1].y, s.v[2].y});
    // Pixels whose centers (k*ONE + HALF) fall inside the bounding box.
    s.x0 = (int)std::max<int64_t>(0, ceil_div(minx - HALF, ONE));
    s.y0 = (int)std::max<int64_t>(0, ceil_div(miny - HALF, ONE));
    s.x1 = (int)std::min<int64_t>(t.w - 1, floor_div(maxx - HALF, ONE));
    s.y1 = (int)std::min<int64_t>(t.h - 1, floor_div(maxy - HALF, ONE));
    return s.x0 <= s.x1 && s.y0 <= s.y1;
}

bool pixel_bounds(const Target& t, const Vertex tri[3], int& x0, int& y0, int& x1, int& y1) {
    Setup s;
    if (!setup(t, tri, false, s)) return false;
    x0 = s.x0; y0 = s.y0; x1 = s.x1; y1 = s.y1;
    return true;
}

// Interpolate attributes from barycentric weights w0+w1+w2 == area (exact
// integers), round to nearest, depth test, write. Shared by both rasterizers so
// any difference between them can only come from coverage or a planted bug.
static inline void shade(Target& t, const Setup& s, int px, int py,
                         int64_t w0, int64_t w1, int64_t w2, bool dut, Stats* st) {
    const int64_t A = s.area;
    const int64_t rnd = (dut && faults.interp_no_round) ? 0 : A / 2;
    auto lerp8 = [&](int c0, int c1, int c2) {
        return (uint32_t)((w0 * c0 + w1 * c1 + w2 * c2 + rnd) / A);
    };
    const __int128 zz = (__int128)w0 * s.v[0].z + (__int128)w1 * s.v[1].z + (__int128)w2 * s.v[2].z;
    const uint32_t z = (uint32_t)((zz + A / 2) / A);
    const int i = py * t.w + px;
    t.hits[i]++;
    if (st) st->pixels_covered++;
    if (t.depth_test && z >= t.depth[i]) return;
    t.depth[i] = z;
    t.color[i] = 0xFF000000u | lerp8(s.v[0].r, s.v[1].r, s.v[2].r) << 16 |
                 lerp8(s.v[0].g, s.v[1].g, s.v[2].g) << 8 | lerp8(s.v[0].b, s.v[1].b, s.v[2].b);
}

void draw_reference(Target& t, const Vertex tri[3], Stats* st) {
    Setup s;
    if (st) st->triangles++;
    if (!setup(t, tri, false, s)) { if (st && s.area == 0) st->degenerate++; return; }
    for (int py = s.y0; py <= s.y1; py++) {
        for (int px = s.x0; px <= s.x1; px++) {
            const int64_t cx = (int64_t)px * ONE + HALF, cy = (int64_t)py * ONE + HALF;
            const int64_t w0 = edge(s.v[1], s.v[2], cx, cy);
            const int64_t w1 = edge(s.v[2], s.v[0], cx, cy);
            const int64_t w2 = edge(s.v[0], s.v[1], cx, cy);
            if (st) st->pixel_tests++;
            if (w0 + s.bias[0] >= 0 && w1 + s.bias[1] >= 0 && w2 + s.bias[2] >= 0)
                shade(t, s, px, py, w0, w1, w2, false, st);
        }
    }
}

void draw_tiled(Target& t, const Vertex tri[3], Stats* st) {
    Setup s;
    if (st) st->triangles++;
    if (!setup(t, tri, true, s)) { if (st && s.area == 0) st->degenerate++; return; }

    const Vertex* e[3][2] = {{&s.v[1], &s.v[2]}, {&s.v[2], &s.v[0]}, {&s.v[0], &s.v[1]}};
    int64_t stepx[3], stepy[3];  // change in edge value per pixel in x and y
    for (int i = 0; i < 3; i++) {
        stepx[i] = -(int64_t)(e[i][1]->y - e[i][0]->y) * ONE;
        stepy[i] =  (int64_t)(e[i][1]->x - e[i][0]->x) * ONE;
    }
    auto eval = [&](int i, int px, int py) {
        return edge(*e[i][0], *e[i][1], (int64_t)px * ONE + HALF, (int64_t)py * ONE + HALF);
    };

    const int tx0 = s.x0 & ~(TILE - 1), ty0 = s.y0 & ~(TILE - 1);
    for (int ty = ty0; ty <= s.y1; ty += TILE) {
        for (int tx = tx0; tx <= s.x1; tx += TILE) {
            // Clip the tile to the triangle's pixel bounds.
            const int rx0 = std::max(tx, s.x0), rx1 = std::min(tx + TILE - 1, s.x1);
            const int ry0 = std::max(ty, s.y0), ry1 = std::min(ty + TILE - 1, s.y1);
            // Edge functions are linear, so their min over the tile is at a corner.
            const int cx[4] = {rx0, rx1, rx0, rx1}, cy[4] = {ry0, ry0, ry1, ry1};
            const int ncheck = (faults.tile_accept_3pt ? 3 : 4);
            bool reject = false, accept = true;
            for (int i = 0; i < 3 && !reject; i++) {
                int outside = 0;
                for (int c = 0; c < 4; c++) {
                    const bool in = eval(i, cx[c], cy[c]) + s.bias[i] >= 0;
                    if (!in) { outside++; if (c < ncheck) accept = false; }
                }
                if (outside == 4) reject = true;
            }
            if (reject) { if (st) st->tiles_rejected++; continue; }

            if (accept) { if (st) st->tiles_accepted++; }
            else        { if (st) st->tiles_partial++;  }

            int64_t row[3] = {eval(0, rx0, ry0), eval(1, rx0, ry0), eval(2, rx0, ry0)};
            for (int py = ry0; py <= ry1; py++) {
                int64_t w[3] = {row[0], row[1], row[2]};
                for (int px = rx0; px <= rx1; px++) {
                    bool in = true;
                    if (!accept) {
                        if (st) st->pixel_tests++;
                        in = w[0] + s.bias[0] >= 0 && w[1] + s.bias[1] >= 0 && w[2] + s.bias[2] >= 0;
                    }
                    if (in) shade(t, s, px, py, w[0], w[1], w[2], true, st);
                    for (int i = 0; i < 3; i++) w[i] += stepx[i];
                }
                for (int i = 0; i < 3; i++) row[i] += stepy[i];
            }
        }
    }
}

}  // namespace raster

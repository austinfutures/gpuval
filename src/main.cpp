// main.cpp — gpuval command line.
//
//   gpuval all        run everything (default)
//   gpuval fp16       half-precision model verification
//   gpuval raster     rasterizer verification
//   gpuval faults     fault-injection campaign: plant bugs, prove the tests catch them
//   gpuval render     draw a 3D scene to render.bmp
//   --quick           smaller vector counts (seconds instead of minutes)
//   --threads N       worker threads for exhaustive checks
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>

#include "fp16.hpp"
#include "raster.hpp"
#include "util.hpp"

static bool all_passed(const std::vector<Check>& v) {
    for (auto& c : v) if (!c.passed()) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Fault campaign. For each planted bug: enable it, run the suites in quick mode,
// record which checks went red and the first failing vector. A bug that no
// check detects is an "escape" and fails the campaign — that is how you measure
// whether a test plan is actually strong enough.
// ---------------------------------------------------------------------------
static bool run_fault_campaign(const Options& base) {
    std::printf("\n== faults: injected-bug campaign (each bug must be caught) ==\n");
    struct Fault { const char* name; const char* what; bool* knob; bool fp; };
    Fault list[] = {
        {"fp16.round_half_up",    "ties round away from zero, not to even",    &fp16::faults.round_half_up,    true},
        {"fp16.flush_subnormals", "tiny results flushed to zero (FTZ)",        &fp16::faults.flush_subnormals, true},
        {"fp16.fma_unfused",      "FMA rounds a*b before adding c",            &fp16::faults.fma_unfused,      true},
        {"raster.no_top_left",    "fill rule ignored on shared edges",         &raster::faults.no_top_left,    false},
        {"raster.tile_accept_3pt","tile trivial-accept skips one corner",      &raster::faults.tile_accept_3pt,false},
        {"raster.interp_no_round","color interpolation truncates",             &raster::faults.interp_no_round,false},
    };
    Options o = base;
    o.quick = true;
    o.verbose = false;
    int escapes = 0;
    for (auto& f : list) {
        *f.knob = true;
        const auto checks = f.fp ? run_fp16_suite(o) : run_raster_suite(o);
        *f.knob = false;
        std::string caught, first;
        for (auto& c : checks) {
            if (c.failures) {
                std::string n = c.name;
                n.erase(std::unique(n.begin(), n.end(), [](char a, char b) { return a == ' ' && b == ' '; }), n.end());
                caught += (caught.empty() ? "" : ", ") + n;
                if (first.empty()) first = c.first_failure;
            }
        }
        const bool ok = !caught.empty();
        if (!ok) escapes++;
        std::printf("  [%s] %-24s %s\n", ok ? "CAUGHT" : "ESCAPE", f.name, f.what);
        if (ok) {
            std::printf("           detected by: %s\n", caught.c_str());
            std::printf("           first vector: %s\n", first.c_str());
        }
    }

    // Debug artifact: what the fill-rule bug looks like on screen.
    {
        Rng g(5);
        raster::Target t(256, 192);
        t.depth_test = false;
        raster::faults.no_top_left = true;
        for (int j = 0; j < 6; j++)
            for (int i = 0; i < 8; i++) {
                auto P = [&](int a, int b) {
                    return raster::Vertex{(a * 32) * raster::ONE + raster::HALF, (b * 32) * raster::ONE + raster::HALF, 0, 0, 0, 0};
                };
                raster::Vertex q[6] = {P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j), P(i + 1, j + 1), P(i, j + 1)};
                raster::draw_tiled(t, q);
                raster::draw_tiled(t, q + 3);
            }
        raster::faults.no_top_left = false;
        for (size_t p = 0; p < t.hits.size(); p++)
            t.color[p] = t.hits[p] == 0 ? 0xFF000000 : t.hits[p] == 1 ? 0xFF404850 : 0xFFFF3030;
        write_bmp("debug_overdraw.bmp", t.w, t.h, t.color);
        std::printf("  wrote debug_overdraw.bmp (red = pixels drawn twice under the fill-rule bug)\n");
    }
    std::printf("  %d of %zu planted bugs caught\n", (int)(sizeof list / sizeof list[0]) - escapes, sizeof list / sizeof list[0]);
    return escapes == 0;
}

// ---------------------------------------------------------------------------
// Render a lit torus through the same rasterizer: float math only for the
// vertex transform, then everything is snapped to fixed point like hardware.
// ---------------------------------------------------------------------------
static void render(const char* path) {
    const int W = 640, H = 480, U = 64, V = 32;
    const double R = 1.0, r = 0.42, ax = 0.9, ay = 0.5;
    std::vector<raster::Vertex> verts((U + 1) * (V + 1));
    for (int i = 0; i <= U; i++) {
        for (int j = 0; j <= V; j++) {
            const double u = 2 * M_PI * i / U, w = 2 * M_PI * j / V;
            double x = (R + r * std::cos(w)) * std::cos(u), y = (R + r * std::cos(w)) * std::sin(u), z = r * std::sin(w);
            double nx = std::cos(w) * std::cos(u), ny = std::cos(w) * std::sin(u), nz = std::sin(w);
            auto rot = [&](double& a, double& b, double t) { double c = std::cos(t), s = std::sin(t), A = a * c - b * s; b = a * s + b * c; a = A; };
            rot(y, z, ax); rot(ny, nz, ax);
            rot(x, z, ay); rot(nx, nz, ay);
            z += 3.2;
            const double f = 0.85 * H / z;
            const double light = std::max(0.0, (nx * -0.4 + ny * -0.6 + nz * -0.7) / 1.0);
            const double base = 0.15 + 0.85 * light;
            raster::Vertex& v = verts[i * (V + 1) + j];
            v.x = (int32_t)std::lround((W / 2 + x * f) * raster::ONE);
            v.y = (int32_t)std::lround((H / 2 + y * f) * raster::ONE);
            v.z = (uint32_t)std::lround((z - 1.5) / 3.5 * 0xFFFFFF);
            v.r = (uint8_t)std::lround(255 * base * (0.55 + 0.45 * std::cos(u)));
            v.g = (uint8_t)std::lround(255 * base * (0.55 + 0.45 * std::cos(u + 2.1)));
            v.b = (uint8_t)std::lround(255 * base * (0.55 + 0.45 * std::cos(u + 4.2)));
        }
    }
    raster::Target t(W, H);
    t.clear(0xFF101418);
    raster::Stats st;
    Timer tm;
    for (int i = 0; i < U; i++) {
        for (int j = 0; j < V; j++) {
            auto at = [&](int a, int b) { return verts[a * (V + 1) + b]; };
            raster::Vertex q[6] = {at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j), at(i + 1, j + 1), at(i, j + 1)};
            raster::draw_tiled(t, q, &st);
            raster::draw_tiled(t, q + 3, &st);
        }
    }
    const double ms = tm.sec() * 1e3;
    write_bmp(path, W, H, t.color);
    std::printf("\n== render ==\n  %s triangles -> %s  (%dx%d, %.1f ms, %s pixels covered)\n",
                commas(st.triangles).c_str(), path, W, H, ms, commas(st.pixels_covered).c_str());
}

int main(int argc, char** argv) {
    Options o;
    std::string cmd = "all";
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--quick")) o.quick = true;
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) o.threads = (unsigned)std::atoi(argv[++i]);
        else cmd = argv[i];
    }
    Timer total;
    bool ok = true;
    if (cmd == "fp16" || cmd == "all") ok &= all_passed(run_fp16_suite(o));
    if (cmd == "raster" || cmd == "all") ok &= all_passed(run_raster_suite(o));
    if (cmd == "faults" || cmd == "all") ok &= run_fault_campaign(o);
    if (cmd == "render" || cmd == "all") render("render.bmp");
    if (cmd != "all" && cmd != "fp16" && cmd != "raster" && cmd != "faults" && cmd != "render") {
        std::printf("usage: gpuval [all|fp16|raster|faults|render] [--quick] [--threads N]\n");
        return 2;
    }
    std::printf("\n%s  (%.1f s)\n", ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED", total.sec());
    return ok ? 0 : 1;
}

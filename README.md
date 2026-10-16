# gpuval — GPU Validation Toolkit (bit-accurate models + rasterizer verification)

A small C++17 project that practices the day-to-day work of a GPU validation /
content engineer: write a **bit-accurate C++ model** of a hardware unit, check it
against **real silicon**, verify a **GPU-style rasterizer** against a golden model,
and prove the test plan works by **injecting bugs** and showing every one is caught.

No GPU needed. Runs on any x86-64 Linux or WSL2 machine.

## What's inside

| Part | What it is | Why a GPU team cares |
|---|---|---|
| `fp16.cpp` | Integer-only model of an IEEE 754 half-precision unit: convert, add, multiply, fused multiply-add. Round-to-nearest-even, subnormals, signed zeros, inf/NaN. | Half precision is the workhorse format for shaders and ML on GPUs. A bit-accurate model is the golden reference RTL is verified against. |
| `reference.cpp` | Independent references: x86 **F16C instructions** (real silicon) and the compiler's `_Float16` soft-float runtime. | Never verify a model against itself. |
| `raster.cpp` | Fixed-point (24.8) triangle rasterizer with the **top-left fill rule**, depth test and attribute interpolation. Two versions: brute-force golden model and an **8x8 tiled** version with trivial reject/accept and incremental edge stepping (the DUT). | This is how hardware rasterizers actually decide coverage. |
| `test_fp16.cpp` | **Exhaustive** verification: all 65,536 halves, all 4,294,967,296 float32 inputs, all 4.29 B operand pairs for add and for mul. Directed corner cases + constrained random for FMA. Multithreaded. | Exhaustive → directed → random is the standard FPU sign-off ladder. |
| `test_raster.cpp` | **Watertightness** (every pixel of a shared-edge mesh covered exactly once), **differential** testing vs the golden model after every triangle, and a **CRC frame signature**. | Cracks/overdraw on shared edges and golden-signature mismatches are classic GPU bring-up bugs. |
| `main.cpp` `faults` | **Fault-injection campaign**: plants 6 realistic bugs (wrong tie rounding, flush-to-zero, unfused FMA, missing fill rule, tile-accept bug, truncating interpolation) and reports which check caught each and the first failing vector. | Measures whether the test plan is strong enough. An uncaught bug is an "escape". |

## Setup (VS Code terminal, Linux or WSL2)

On Windows, open VS Code in WSL (bottom-left `><` button → "Connect to WSL"), then:

```bash
sudo apt update && sudo apt install -y build-essential cmake
```

## Build and run

```bash
cd gpuval
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/gpuval --quick      # everything, small vector counts (~5 s)
./build/gpuval              # everything, full exhaustive (~1–4 min depending on cores)
```

Individual pieces:

```bash
./build/gpuval fp16         # half-precision model, exhaustive
./build/gpuval raster       # rasterizer: watertight + differential + signature
./build/gpuval faults       # bug-injection campaign, writes debug_overdraw.bmp
./build/gpuval render       # draws a lit torus to render.bmp
```

Open `render.bmp` and `debug_overdraw.bmp` in VS Code to view them.

## Results (2-core cloud VM; faster on a laptop)

```
[PASS] to_float   exhaustive         65,536 vectors      0 fails
[PASS] from_float exhaustive  4,294,967,296 vectors      0 fails   (vs x86 F16C)
[PASS] add        exhaustive  4,294,967,296 vectors      0 fails
[PASS] mul        exhaustive  4,294,967,296 vectors      0 fails
[PASS] fma        random         67,108,864 vectors      0 fails
[PASS] watertight meshes          1,785,620 pixels       0 fails
[PASS] differential vs golden   208,695,230 pixel cmps   0 fails
tiles: 57.8% trivially rejected, 39.1% trivially accepted, 3.1% partial
per-pixel edge tests: 97.2% fewer than brute force
6 of 6 planted bugs caught
```

A favorite catch from the fault campaign: with the unfused-FMA bug,
`fma(-2^-24, 2^-24, +0)` returns `+0` instead of `-0`. Rounding the tiny product
first throws away its sign — exactly the kind of one-bit difference a
bit-accurate model exists to pin down.

## Ideas to extend

- Add a Vulkan or Metal compute shader that runs the same fp16 vectors on a real GPU and compares to the model.
- Add perspective-correct interpolation and a texture sampler (bilinear filtering in fixed point).
- Add a bf16 or fp8 (E4M3/E5M2) model reusing `round_pack()`.

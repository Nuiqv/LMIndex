# LMIndex

## Overview
LMIndex is a new learned indexing architecture that delivers fast point & range queries, low memory footprint, and robust performance under updates. It combines a constant-time mapping layer with piecewise linear models to support high-performance indexing.

LMIndex with Gaps(LMG) extends LMIndex with a gapped array layout and adaptive correction mechanism, providing significantly better performance in mixed read-write workloads.

## Build
Prerequisites:
- CMake ≥ 3.12
- A C++17-compliant compiler (GCC 9+, Clang 10+, MSVC 19.28+)

Build commands:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target LMD SIMPLE_TEST
```
Artifacts are placed in `bin/`. To build a debug version, pass `-DCMAKE_BUILD_TYPE=Debug`.

## Quick Start
Run the sample program to see the core operations on a random dataset:
```bash
./bin/SIMPLE_TEST
```
It performs a point query, range query, insert, and delete against `LM_Index_Gaps`.

## License
Distributed under the MIT License; see `LICENSE`.

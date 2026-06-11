#!/bin/bash
set -e

# Configuration
CFLAGS="-O2 -march=native -fopenmp -msse2 -Wall"
CXXFLAGS="-O2 -march=native -fopenmp -msse2 -Wall"
LDFLAGS="-lm -fopenmp -lgmp -lstdc++"

# Source files (as listed in digital_twin.c header)
C_SOURCES=(
    quhit_core.c
    quhit_gates.c
    quhit_measure.c
    quhit_entangle.c
    quhit_register.c
    quhit_substrate.c
    quhit_triality.c
    quhit_triadic.c
    quhit_lazy.c
    quhit_calibrate.c
    quhit_dyn_integrate.c
    quhit_peps_grow.c
    quhit_hexagram.c
    quhit_svd_gate.c
    s6_exotic.c
    bigint.c
    mps_overlay.c
    peps_overlay.c
    peps3d_overlay.c
    peps4d_overlay.c
    peps5d_overlay.c
    peps6d_overlay.c
)

# Compile C files
echo "Compiling C sources..."
for src in "${C_SOURCES[@]}"; do
    obj="${src%.c}.o"
    echo "  gcc -c $src -> $obj"
    gcc $CFLAGS -c "$src" -o "$obj"
done

# Compile Digital Twin main
echo "Compiling digital_twin.c..."
gcc $CFLAGS -c digital_twin.c -o digital_twin.o

# Link everything
echo "Linking digital_twin..."
gcc digital_twin.o \
    quhit_core.o quhit_gates.o quhit_measure.o quhit_entangle.o \
    quhit_register.o quhit_substrate.o quhit_triality.o \
    quhit_triadic.o quhit_lazy.o quhit_calibrate.o \
    quhit_dyn_integrate.o quhit_peps_grow.o quhit_hexagram.o \
    quhit_svd_gate.o s6_exotic.o bigint.o \
    mps_overlay.o peps_overlay.o peps3d_overlay.o peps4d_overlay.o \
    peps5d_overlay.o peps6d_overlay.o \
    $LDFLAGS -o digital_twin

echo "Build successful: ./digital_twin"

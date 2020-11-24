#!/bin/bash -x

# for bluegene
# mpixlcxx_r -qsmp=omp main.cpp -o prog

# for polus
module load SpectrumMPI
module load OpenMPI
mpixlC -qsmp=omp main.cpp -o prog
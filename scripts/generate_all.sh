#!/bin/bash

export PYOPENCL_CTX=port
export SUMPY_FORCE_SYMBOLIC_BACKEND=sympy

python scripts/compute_m2m_error.py "generate(LaplaceKernel(2))"
python scripts/compute_m2m_error.py "generate(LaplaceKernel(3))"
python scripts/compute_m2m_error.py "generate(HelmholtzKernel(2))"
python scripts/compute_m2m_error.py "generate(HelmholtzKernel(3))"
python scripts/compute_m2m_error.py "generate(BiharmonicKernel(2))"
python scripts/compute_m2m_error.py "generate(BiharmonicKernel(3))"

python scripts/compute_flop_count.py "generate(LaplaceKernel(2))"
python scripts/compute_flop_count.py "generate(LaplaceKernel(3))"

python scripts/compute_ie_error.py

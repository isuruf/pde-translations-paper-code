#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

MINIFORGE_HOME=${MINIFORGE_HOME:-${HOME}/.miniforge}
source "$MINIFORGE_HOME/etc/profile.d/conda.sh"
conda activate "$MINIFORGE_HOME/envs/paper"

export PYOPENCL_CTX=${PYOPENCL_CTX:-0:0}
export MPLBACKEND=${MPLBACKEND:-Agg}
export SUMPY_FORCE_SYMBOLIC_BACKEND=sympy

mkdir -p data
cd data

python ../scripts/compute_flop_count.py "generate(LaplaceKernel(2))"
python ../scripts/compute_flop_count.py "generate(LaplaceKernel(3))"

python ../scripts/compute_m2m_error.py "generate(LaplaceKernel(2))"
python ../scripts/compute_m2m_error.py "generate(LaplaceKernel(3))"
python ../scripts/compute_m2m_error.py "generate(HelmholtzKernel(2))"
python ../scripts/compute_m2m_error.py "generate(HelmholtzKernel(3))"
python ../scripts/compute_m2m_error.py "generate(BiharmonicKernel(2))"
python ../scripts/compute_m2m_error.py "generate(BiharmonicKernel(3))"
python ../scripts/compute_m2m_error.py "generate(HelmholtzKernel(2), False)"
python ../scripts/compute_m2m_error.py "generate(HelmholtzKernel(3), False)"

python ../scripts/compute_m2m_error_vs_k.py "generate(HelmholtzKernel(2))"
python ../scripts/compute_m2m_error_vs_k.py "generate(HelmholtzKernel(3))"

python ../scripts/compute_l2l_error.py "generate(LaplaceKernel(2))"
python ../scripts/compute_l2l_error.py "generate(LaplaceKernel(3))"
python ../scripts/compute_l2l_error.py "generate(HelmholtzKernel(2))"
python ../scripts/compute_l2l_error.py "generate(HelmholtzKernel(3))"
python ../scripts/compute_l2l_error.py "generate(BiharmonicKernel(2))"
python ../scripts/compute_l2l_error.py "generate(BiharmonicKernel(3))"

python ../scripts/compute_heat_m2m_error.py
python ../scripts/compute_heat_l2l_error.py
python ../scripts/compute_heat_m2l_error.py

python ../scripts/compute_ie_error.py

cd ..
conda activate "$MINIFORGE_HOME/envs/paper-plot"

mkdir -p figures

python scripts/plot_m2m_error.py
python scripts/plot_m2m_error_compare.py
python scripts/plot_m2m_error_vs_k.py
python scripts/plot_l2l_error.py
python scripts/plot_heat_error.py
python scripts/plot_flop_count.py
python scripts/latex_table_ie_error.py

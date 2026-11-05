# Automatic Synthesis of Low-Complexity Translation Operators for the Fast Multipole Method

Isuru Fernando, Chaoqi Lin and Andreas Kloeckner

[Arxiv](https://arxiv.org/abs/2305.17867)

This repository contains numerical experiments and Lean code verifying the theoretical
claims of the paper.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6638926.svg)](https://doi.org/10.5281/zenodo.6638926)

### Install locally

    ./install.sh
    ./run.sh

The installer creates `paper` for computation and `paper-plot` for plotting
under `~/.miniforge`. Plotting runs in a separate environment because
tikzplotlib 0.10.1 needs an older matplotlib. The figures reproduce the
paper's plotted data; tick spacing and label formatting may differ.
Set `MINIFORGE_HOME` to use a different installation directory.
If either environment already exists, `install.sh` stops; set `MINIFORGE_HOME`
to a fresh directory.
The default OpenCL device is `0:0`; override `PYOPENCL_CTX` to select another.

### Run Lean verification

    cd lean
    lake exe cache get
    lake build

### Build Docker image

    docker build -t pde-translations-paper-code .

### Run experiments

    container="pde-translations-$(date +%s)"
    docker run --name "$container" pde-translations-paper-code

    docker cp "$container":/home/paper/pde-translations-results/data data
    docker cp "$container":/home/paper/pde-translations-results/figures figures

The platform package lists in `envs/` are generated from `envs/conda.yml` and
`envs/plot.yml`
with `bash envs/generate_files.sh`. Python library revisions are pinned in
`install.sh`.

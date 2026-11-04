# Automatic Synthesis of Low-Complexity Translation Operators for the Fast Multipole Method

Isuru Fernando, Shawn Lin and Andreas Kloeckner

[Arxiv](https://arxiv.org/abs/2305.17867)

This repository contains numerical experiments and Lean code verifying the theoretical
claims of the paper.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6638926.svg)](https://doi.org/10.5281/zenodo.6638926)

### Run Lean verification

    cd lean
    lake exe cache get
    lean build

### Build Docker image

    docker build -f Dockerfile . -t pde-translations-paper-code

### Run experiments

    container="pde-translations-$(date +%s)"
    docker run -it --name $container pde-translations-paper-code /bin/bash run.sh

    docker cp $container:/home/paper/pde-translations-results/data data
    docker cp $container:/home/paper/pde-translations-results/figures figures

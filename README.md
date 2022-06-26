[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6638926.svg)](https://doi.org/10.5281/zenodo.6638926)

### Build Docker image

    docker build -f Dockerfile . -t pde-translations-paper-code

### Run experiments

    container="pde-translations-$(date +%s)"
    docker run -it --name $container pde-translations-paper-code /bin/bash run.sh

    docker cp $container:/home/paper/pde-translations-results/data data
    docker cp $container:/home/paper/pde-translations-results/figures figures

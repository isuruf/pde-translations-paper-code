### Build Docker image

    docker build -f Dockerfile . -t pde-translations-paper-code

### Run experiments

    container="pde-translations-$(date +%s)"
    docker run -it --name $container pde-translations-paper-code /bin/bash run.sh

    docker cp $container:/home/paper/pde-translations-results/data data
    docker cp $container:/home/paper/pde-translations-results/figures figures

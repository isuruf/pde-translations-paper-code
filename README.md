### Build Docker image

    docker build -f Dockerfile . -t pde-translations-paper-code

### Run experiments

    mkdir data
    mkdir figures
    docker run -it -v \
        $PWD/figures:/home/paper/pde-translations-results/figures \
        $PWD/data:/home/paper/pde-translations-results/data \
        pde-translations-paper-code \
        /bin/bash run.sh

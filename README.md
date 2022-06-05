### Build Docker image

    docker build -f Dockerfile . -t pde-translations-paper-code

### Run experiments

    docker run -it -v $PWD:/home/paper/pde-translations-results pde-translations-paper-code /bin/bash run.sh

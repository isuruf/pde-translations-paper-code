FROM debian:bullseye
LABEL maintainer="idf2@illinois.edu"

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash paper
USER paper

# Install latexrun script
RUN mkdir /home/paper/bin
WORKDIR /home/paper/bin

# Install Conda environment
RUN mkdir /home/paper/pde-translations-results
WORKDIR /home/paper/pde-translations-results
COPY --chown=paper envs envs
COPY --chown=paper install.sh install.sh
RUN ./install.sh \
    && /home/paper/.miniforge/bin/conda clean --all --yes

# Copy the scripts for running the experiments
COPY --chown=paper scripts scripts
COPY --chown=paper run.sh run.sh

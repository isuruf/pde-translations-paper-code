FROM debian:bookworm-slim
LABEL maintainer="idf2@illinois.edu"

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash paper
USER paper
RUN mkdir -p /home/paper/pde-translations-results
WORKDIR /home/paper/pde-translations-results

COPY --chown=paper envs envs
COPY --chown=paper install.sh install.sh
RUN ./install.sh \
    && /home/paper/.miniforge/bin/conda clean --all --yes

COPY --chown=paper scripts scripts
COPY --chown=paper run.sh run.sh
ENV PYOPENCL_CTX=0:0 MPLBACKEND=Agg SUMPY_FORCE_SYMBOLIC_BACKEND=sympy
CMD ["/bin/bash", "run.sh"]

#!/bin/bash

MINIFORGE_HOME=${HOME}/.miniforge

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/4.12.0-2"
MINIFORGE_FILE="Mambaforge-$(uname)-$(uname -m).sh"
curl -L -O "${MINIFORGE_URL}/${MINIFORGE_FILE}"
rm -rf ${MINIFORGE_HOME}
bash $MINIFORGE_FILE -b -p ${MINIFORGE_HOME}

source ${MINIFORGE_HOME}/bin/activate

mamba create --yes --name paper --file "envs/$(uname)-$(uname -m).txt"

source activate paper

pip install \
    "git+https://github.com/inducer/loopy.git@0239627c15bc2528291a008b3ae47f064b99da86#egg=loopy" \
    "git+https://github.com/inducer/arraycontext.git@8d7c872ea33f5e6f6251fc9e00c6944f8d15f3a9#egg=arraycontext" \
    "git+https://github.com/inducer/boxtree.git@e5c1d88c5eec4cb2cc7e466961d126b9c58f853c#egg=boxtree" \
    "git+https://github.com/inducer/gmsh_interop.git@bb219ca1173512d741b9ce1cb2e3dee4179e78c1#egg=gmsh_interop" \
    recursivenodes==0.2.0 \
    "git+https://github.com/inducer/meshmode.git@c798a6d14e1a7feb13efea52a1cc49456d80e48f#egg=meshmode" \
    "git+https://github.com/inducer/sumpy.git@dfc0e71609e6590779cbe320bf75ffb1c935916e#egg=sumpy" \
    "git+https://github.com/inducer/pytential.git@b535d0ea869a53e8dd8ff4f9162d56fa19b4c66a#egg=pytential"



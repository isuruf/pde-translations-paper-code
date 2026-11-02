#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

MINIFORGE_HOME=${MINIFORGE_HOME:-${HOME}/.miniforge}
MINIFORGE_VERSION=26.9.4-1

if [[ ! -x "$MINIFORGE_HOME/bin/conda" ]]; then
    installer_os=$(uname -s)
    if [[ "$installer_os" == Darwin ]]; then
        installer_os=MacOSX
    fi
    installer_dir=$(mktemp -d)
    installer="$installer_dir/Miniforge3.sh"
    trap 'rm -rf "$installer_dir"' EXIT
    curl --fail --location --retry 3 --silent --show-error \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${installer_os}-$(uname -m).sh" \
        --output "$installer"
    bash "$installer" -b -p "$MINIFORGE_HOME"
fi

source "$MINIFORGE_HOME/etc/profile.d/conda.sh"
for name in paper paper-plot; do
    if [[ -e "$MINIFORGE_HOME/envs/$name" ]]; then
        echo "Environment $MINIFORGE_HOME/envs/$name already exists; use a fresh MINIFORGE_HOME." >&2
        exit 1
    fi
done

platform="$(uname -s)-$(uname -m)"
conda create --yes --prefix "$MINIFORGE_HOME/envs/paper" --file "envs/$platform.txt"
conda create --yes --prefix "$MINIFORGE_HOME/envs/paper-plot" --file "envs/$platform-plot.txt"
conda activate "$MINIFORGE_HOME/envs/paper"

python -m pip install \
    "git+https://github.com/inducer/loopy.git@476a6e1a6e3dd27a6cbaac467439de25638050e2" \
    "git+https://github.com/inducer/arraycontext.git@94cff7dd0735793c0d6534211059562cb0765b06" \
    "git+https://github.com/inducer/boxtree.git@5b9c44886fe47236a0aae78bc1989fcca9da32ef" \
    "git+https://github.com/inducer/gmsh_interop.git@586f6696ddff7ebe97282bbdd7750c7b8b4e3bcf" \
    "git+https://github.com/inducer/meshmode.git@bbaf4ab4ecfb3b1cfcdac79165d4ca61fe0de64a" \
    "git+https://github.com/inducer/sumpy.git@c435f826adfaa38404c7389fb4c737bf558cb581" \
    "git+https://github.com/inducer/pytential.git@8700a2918085572ae86ace15a0beb2f106458bb2" \
    "git+https://github.com/inducer/modepy.git@01ac7955d30e248e460ccd751d40da4b0ff3e3e6"

python -m pip install --force-reinstall --no-deps \
    "git+https://github.com/inducer/pymbolic.git@6e66a223597b9b0f719c7cac1b4f59f95e651b64"
python -m pip check

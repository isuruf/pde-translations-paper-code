"""
Run this script using gitlab.tiker.net/isuruf/sumpy branch m2m
Also need https://gitlab.tiker.net/inducer/sumpy/merge_requests/113
"""

import pyopencl as cl
import sumpy.toys as t
import numpy as np
import numpy.linalg as la
from sumpy.kernel import (
    HelmholtzKernel,
    LaplaceKernel,  # noqa: F401
    BiharmonicKernel,
    HeatKernel,
)
import sys
import json
from functools import partial

from sumpy.expansion.multipole import (
    VolumeTaylorMultipoleExpansion,
    LinearPDEConformingVolumeTaylorMultipoleExpansion,
)
from sumpy.expansion.local import (
    VolumeTaylorLocalExpansion,
    LinearPDEConformingVolumeTaylorLocalExpansion,
)
from sumpy.expansion.m2l import (
    NonFFTM2LTranslationClassFactory,
    FFTM2LTranslationClassFactory,
    VolumeTaylorM2LWithPreprocessedMultipoles,
)
from sumpy.array_context import _acf
from pytools.obj_array import make_obj_array
from sumpy.visualization import FieldPlotter


def generate(knl):
    extra_kernel_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kernel_kwargs = {"k": 5}

    dim = knl.dim

    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
    if isinstance(knl, HeatKernel):
        extra_kwargs["alpha"] = 1

    actx = _acf()
    target_kernels = [knl]
    data = []

    origin = np.array([0, 0, 0, 0.5][-knl.dim :], np.float64)
    ntargets_per_dim = 6
    nsources_per_dim = 11

    sources_grid = np.meshgrid(
        *[np.linspace(0, 1, nsources_per_dim) for _ in range(dim)]
    )
    sources_grid = np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1) - 0.5
    nsources = nsources_per_dim**dim

    strengths = np.ones(nsources, dtype=np.float64) * (1 / nsources)

    targets_grid = np.meshgrid(
        *[np.linspace(0, 1, ntargets_per_dim) for _ in range(dim)]
    )
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1) - 0.5
    ntargets = ntargets_per_dim**dim

    toy_ctx = t.ToyContext(
        actx.context,
        kernel=knl,
        extra_kernel_kwargs=extra_kwargs,
        m2l_use_fft=False,
    )

    m1_rscale = 0.5
    m2_rscale = 0.25
    l1_rscale = 0.5
    l2_rscale = 0.25

    def norm(x):
        return np.max(np.abs(x))

    all_data = []
    for order in range(2, 13, 2):
        data = {"order": order, "h": [], "error": []}
        all_data.append(data)
        for h in 2.0 ** np.arange(-4, -13, -1):
            sources = h * sources_grid + origin[:, np.newaxis]
            src_size = h
            p = t.PointSources(toy_ctx, sources, weights=strengths)
            targets = (origin + np.array([0, 0, 0, 0.5][-knl.dim :]))[
                :, np.newaxis
            ] + 0.1 * (targets_grid - 0.5)

            mpole_center = origin + np.array([0, 0, 0, h / 2][-knl.dim :])
            p2m = t.multipole_expand(p, mpole_center, order, m1_rscale)
            p2p = p.eval(targets)
            p2m2p = p2m.eval(targets)
            p2m2m = t.multipole_expand(
                p2m, origin + np.array([h, h, h, h][: -knl.dim]), order, m1_rscale
            )
            p2m2m2p = p2m2m.eval(targets)
            err = norm((p2m2m2p - p2p) / p2p)

            diff = np.linalg.norm(sources - origin[:, np.newaxis], axis=0)
            conv_factor = norm(diff)
            data["h"].append(conv_factor)
            data["error"].append(err)
            print(data["h"][-1], data["error"][-1], order)
            name = type(knl).__name__
            with open(f"data/{name}_{dim - 1}D_m2m_error.json", "w") as f:
                json.dump(all_data, f, indent=2)


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(HeatKernel(1))

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
        extra_kwargs["alpha"] = 0.1

    actx = _acf()
    target_kernels = [knl]

    origin = np.array([0, 0, 0, 0.5][-knl.dim :], np.float64)
    ntargets_per_dim = 3
    nsources_per_dim = 11

    sources_grid = np.meshgrid(
        *[np.linspace(0, 1, nsources_per_dim) for _ in range(dim)]
    )
    sources_grid = np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1) - 0.5
    sources_grid[-1, :] = 0.0
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
        m2l_use_fft=True,
    )

    def norm(x):
        return np.max(np.abs(x))

    for algo in ("l2l", "m2l"):
        all_data = []
        for order in range(2, 13, 2):
            data = {"h": [], "order": order, "error": []}
            all_data.append(data)
            for h in 2.0 ** np.arange(-2, -13, -1):
                src_size = 1
                sources = src_size * sources_grid + origin[:, np.newaxis]
                p = t.PointSources(toy_ctx, sources, weights=strengths)
                mpole_center = origin + np.array([0, 0, 0, 0.0][-knl.dim :])
                local_center1 = origin + np.array([0, 0, 0, 1.0 - h][-knl.dim :])
                local_center2 = origin + np.array([0, 0, 0, 1.0][-knl.dim :])
                targets = local_center2[:, np.newaxis] + h * (targets_grid - 0.5)

                diff = norm(local_center2 - mpole_center)
                m1_rscale = l1_rscale = l2_rscale = m2_rscale = diff / order

                p2p = p.eval(targets)
                if algo == "m2l":
                    p2m = t.multipole_expand(p, mpole_center, order, m1_rscale)
                    p2m2l = t.local_expand(p2m, local_center2, order, l1_rscale)
                    p2m2l2p = p2m2l.eval(targets)
                    p2m2p = p2m.eval(targets)
                    err = norm((p2m2l2p - p2m2p) / p2m2p)
                elif algo == "l2l":
                    p2l = t.local_expand(p, local_center1, order, l1_rscale)
                    p2l2l = t.local_expand(p2l, local_center2, order, l2_rscale)
                    p2l2l2p = p2l2l.eval(targets)
                    p2l2p = p2l.eval(targets)
                    err = norm((p2l2l2p - p2l2p) / p2l2p)

                dist = np.linalg.norm(targets - local_center2[:, np.newaxis], axis=0)
                conv_factor = norm(dist)

                data["h"] += [norm(conv_factor)]
                data["error"] += [err]
                print(norm(conv_factor), err, order)
                name = type(knl).__name__
                with open(f"data/{name}_{dim - 1}D_{algo}_error.json", "w") as f:
                    json.dump(all_data, f, indent=2)


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(HeatKernel(1))

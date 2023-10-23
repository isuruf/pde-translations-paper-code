"""
Run this script using gitlab.tiker.net/isuruf/sumpy branch m2m
Also need https://gitlab.tiker.net/inducer/sumpy/merge_requests/113
"""

import pyopencl as cl
import sumpy.toys as t
import numpy as np
import numpy.linalg as la
from sumpy.kernel import (HelmholtzKernel, LaplaceKernel,  # noqa: F401
                          BiharmonicKernel, HeatKernel)
import sys
import json
from functools import partial

from sumpy.expansion.multipole import (
        VolumeTaylorMultipoleExpansion,
        LinearPDEConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion,
        LinearPDEConformingVolumeTaylorLocalExpansion)
from sumpy.expansion.m2l import (NonFFTM2LTranslationClassFactory,
        FFTM2LTranslationClassFactory, VolumeTaylorM2LWithPreprocessedMultipoles)
from sumpy.array_context import _acf
from pytools.obj_array import make_obj_array
from sumpy.visualization import FieldPlotter    


def generate(knl):
    extra_kernel_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kernel_kwargs = {'k': 5}

    dim = knl.dim

    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
    if isinstance(knl, HeatKernel):
        extra_kwargs["alpha"] = 0.1

    actx = _acf()
    target_kernels = [knl]

    origin = np.array([0, 0, 0, 0][-knl.dim:], np.float64)
    ntargets_per_dim = 3
    nsources_per_dim = 10

    sources_grid = np.meshgrid(*[np.linspace(0, 1, nsources_per_dim)
                                 for _ in range(dim)])
    sources_grid = (np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1) - 0.5)
    sources_grid[-1, :] = 0.0
    nsources = nsources_per_dim**dim

    strengths = np.ones(nsources, dtype=np.float64) * (1/nsources)

    targets_grid = np.meshgrid(*[np.linspace(0, 1, ntargets_per_dim)
                                 for _ in range(dim)])
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1) - 0.5
    ntargets = ntargets_per_dim**dim

    toy_ctx = t.ToyContext(
        actx.context,
        kernel=knl,
        extra_kernel_kwargs=extra_kwargs,
        m2l_use_fft=True,
    )

    order = 5

    def norm(x):
        return np.max(np.abs(x))

    for algo in ("p2l2p", "p2m2l2p"):
        data = []
        for h in 2.0**np.arange(-2, -13, -1):
            src_size = 1
            m1_rscale = l1_rscale = src_size / 8
            sources = src_size * sources_grid + origin[:, np.newaxis]
            p = t.PointSources(toy_ctx, sources, weights=strengths)
            mpole_center = origin + np.array([0, 0, 0, 0.0][-knl.dim:])
            local_center = origin + np.array([0, 0, 0, 0.5][-knl.dim:])
            targets = local_center[:, np.newaxis] + h*(targets_grid - 0.5)

            p2p = p.eval(targets)
            if algo == "p2m2l2p":
                p2m = t.multipole_expand(p, mpole_center, order, m1_rscale)
                p2m2l = t.local_expand(p2m, local_center, order, l1_rscale)
                p2m2l2p = p2m2l.eval(targets)
                p2m2p = p2m.eval(targets)
                err = norm((p2m2l2p - p2m2p)/p2m2p)
            elif algo == "p2l2p":
                p2l = t.local_expand(p, local_center, order, l1_rscale)
                p2l2p = p2l.eval(targets)
                p2p = p.eval(targets)
                err = norm((p2l2p - p2p)/p2p)

            dist = targets - local_center[:, np.newaxis]
            conv_factor = np.linalg.norm(dist)
            error_model = conv_factor**(order+1)
            ratio = norm(err/error_model)
            
            data.append({"h": norm(conv_factor), "order": order, "error": err, "ratio": ratio})
            print(data[-1])
            name = type(knl).__name__
            with open(f'{name}_{dim - 1}D_{algo}_error.json', 'w') as f:
                json.dump(data, f, indent=2)

        print(data)

        if 1:
            import matplotlib.pyplot as plt
            x = [d['h'] for d in data]
            y = [d['error'] for d in data]
            plt.loglog(x, y, label="Error", marker="o")
            plt.loglog(x, np.array(x)**(order + 1)*y[0]/x[0]**(order + 1), label="error_model")
            plt.gca().set_ylim(bottom=3e-17)
            plt.xlabel("convergence factor")
            plt.ylabel("error")
            plt.title(f"{algo.upper()} error for Heat 1D")
            plt.legend()
            plt.show()


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(HeatKernel(1))

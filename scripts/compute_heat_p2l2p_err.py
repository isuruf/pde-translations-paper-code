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
        extra_kwargs["alpha"] = 1

    actx = _acf()
    target_kernels = [knl]
    data = []

    origin = np.array([0, 0, 0, 0][-knl.dim:], np.float64)
    ntargets_per_dim = 1
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
        m2l_use_fft=False,
    )


    m1_rscale = 0.5
    m2_rscale = 0.25
    l1_rscale = 0.5
    l2_rscale = 0.25

    order = 5

    def norm(x):
        return np.max(np.abs(x))

    # for h in [0.475, 0.45, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]:
    for h in 2.0**np.arange(-2, -12, -1):
        src_size = 1
        sources = src_size * sources_grid + origin[:, np.newaxis]
        p = t.PointSources(toy_ctx, sources, weights=strengths)
        center = origin + np.array([0, 0, 0, 0.5][-knl.dim:])
        targets = center[:, np.newaxis] + h*(targets_grid - 0.5)

        p2p = p.eval(targets)
        p2l = t.local_expand(p, center, order, l1_rscale)
        p2l2p = p2l.eval(targets)
        err = norm((p2l2p - p2p)/p2p)
        
        dist = targets - center[:, np.newaxis]
        conv_factor = np.linalg.norm(dist)
        error_model = conv_factor**(order+1)
        ratio = norm(err/error_model)
        
        data.append({"h": norm(conv_factor), "order": order, "error": err, "ratio": ratio})
        print(data[-1])
        name = type(knl).__name__
        with open(f'{name}_{dim - 1}D_p2l2p_error.json', 'w') as f:
            json.dump(data, f, indent=2)

    print(data)

    if 1:
        import matplotlib.pyplot as plt
        x = [d['h'] for d in data]
        y = [d['error'] for d in data]
        plt.loglog(x, y, label="Error", marker="o")
        plt.loglog(x, np.array(x)**(order + 1)*y[0]/x[0]**(order + 1), label="error_model")
        plt.xlabel("convergence factor")
        plt.ylabel("error")
        plt.title("P2L2P error for Heat 1D")
        plt.legend()
        plt.show()


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(HeatKernel(1))

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

    mpole_expn_class = LinearPDEConformingVolumeTaylorMultipoleExpansion
    local_expn_class = LinearPDEConformingVolumeTaylorLocalExpansion
    
    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
    if isinstance(knl, HeatKernel):
        extra_kwargs["alpha"] = 0.1

    actx = _acf()
    target_kernels = [knl]
    data = []
    eval_offset = np.array([0.0, 0.0, 0.0, 9.5][-knl.dim:])

    origin = np.array([0, 0, 1, 2][-knl.dim:], np.float64)
    ntargets_per_dim = 4
    nsources_per_dim = 3

    sources_grid = np.meshgrid(*[np.linspace(0, 1, nsources_per_dim)
                                 for _ in range(dim)])
    sources_grid = np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1)
    sources = (-0.5 + sources_grid) + origin[:, np.newaxis]
    nsources = nsources_per_dim**dim

    strengths = np.ones(nsources, dtype=np.float64) * (1/nsources)

    targets_grid = np.meshgrid(*[np.linspace(0, 1, ntargets_per_dim)
                                 for _ in range(dim)])
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1)
    targets = eval_offset[:, np.newaxis] + 0.25 * (targets_grid - 0.5)
    ntargets = ntargets_per_dim**dim

    centers = (np.array(
            [
                # box 0: particles, first mpole here
                [0, 0, 0, 0][-knl.dim:],

                # box 1: second mpole here
                np.array([0.1, 0, 0, 0][:knl.dim], np.float64),

                # box 2: first local here
                eval_offset + np.array([0, 0.1, 0.2, 0.5][-knl.dim:], np.float64),

                # box 3: second local and eval here
                eval_offset
                ],
            dtype=np.float64) + origin).T.copy()

    del eval_offset

    if dim == 2:
        orders = list(range(2, 25, 2))
    else:
        orders = list(range(2, 13, 2))

    m2l_factory = FFTM2LTranslationClassFactory()
    m2l_translation = m2l_factory.get_m2l_translation_class(knl, local_expn_class)()

    toy_ctx = t.ToyContext(
            actx.context,
            kernel=knl,
            local_expn_class=partial(local_expn_class,
                m2l_translation=m2l_translation),
            mpole_expn_class=mpole_expn_class,
            extra_kernel_kwargs=extra_kwargs,
    )

    p = t.PointSources(toy_ctx, sources, weights=strengths)
    p2p = p.eval(targets)

    m1_rscale = 0.5
    m2_rscale = 0.25
    l1_rscale = 0.5
    l2_rscale = 0.25

    for order in orders:
        p2m = t.multipole_expand(p, centers[:, 0], order, m1_rscale)
        p2m2p = p2m.eval(targets)
        err = la.norm(p2m2p - p2p)/la.norm(p2p)

        p2m2m = t.multipole_expand(p2m, centers[:, 1], order, m2_rscale)
        p2m2m2p = p2m2m.eval(targets)
        err = la.norm(p2m2m2p - p2p)/la.norm(p2p)

        p2m2m2l = t.local_expand(p2m2m, centers[:, 2], order, l1_rscale)
        p2m2m2l2p = p2m2m2l.eval(targets)
        err = la.norm(p2m2m2l2p - p2p)/la.norm(p2p)

        p2m2m2l2l = t.local_expand(p2m2m2l, centers[:, 3], order, l2_rscale)
        p2m2m2l2l2p = p2m2m2l2l.eval(targets)
        err = la.norm(p2m2m2l2l2p - p2p)/la.norm(p2p)
        data.append({"order": order, "error": err})
        print(data)
        
        name = type(knl).__name__
        with open(f'{name}_{dim - 1}D_p2m2m2l2lp_error.json', 'w') as f:
            json.dump(data, f, indent=2)


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(HeatKernel(1))

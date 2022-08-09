"""
Run this script using gitlab.tiker.net/isuruf/sumpy branch m2m
Also need https://gitlab.tiker.net/inducer/sumpy/merge_requests/113
"""

import pyopencl as cl
import sumpy.toys as t
import numpy as np
from sumpy.kernel import (HelmholtzKernel, LaplaceKernel,  # noqa: F401
                          BiharmonicKernel)
import sys

from sumpy.expansion.multipole import (
        VolumeTaylorMultipoleExpansion,
        LinearPDEConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion,
        LinearPDEConformingVolumeTaylorLocalExpansion)


def generate(knl):
    extra_kernel_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kernel_kwargs = {'k': 5}

    dim = knl.dim

    mpole_expn_classes = [LinearPDEConformingVolumeTaylorMultipoleExpansion,
                          VolumeTaylorMultipoleExpansion]
    local_expn_classes = [LinearPDEConformingVolumeTaylorLocalExpansion,
                          VolumeTaylorLocalExpansion]

    eval_center = np.array([0.1, 0.2, 0.3][:dim]).reshape(dim, 1)

    ntargets_per_dim = 50
    nsources_per_dim = 50

    sources_grid = np.meshgrid(*[np.linspace(0, 1, nsources_per_dim)
                                 for _ in range(dim)])
    sources_grid = np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1)

    targets_grid = np.meshgrid(*[np.linspace(0, 1, ntargets_per_dim)
                                 for _ in range(dim)])
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1)

    targets = eval_center - 0.1*(targets_grid - 0.5)

    ctx = cl.create_some_context()
    data = []
    for order in range(2, 13, 2):
        print(order)
        h_values = 2.0**np.arange(-10, -3)
        distances = []
        errs = []
        for h in h_values:
            mpole_center = np.array([h, h, h][:dim]).reshape(dim, 1)
            sources = (h*(-0.5+sources_grid.astype(np.float64)) + mpole_center)
            second_center = mpole_center - h
            r1 = np.max(np.linalg.norm(sources - mpole_center, axis=0))
            r2 = r1 + np.linalg.norm(second_center - mpole_center)
            distances.append(r2)
            m2m_vals = [0, 0]
            for i, (mpole_expn_class, local_expn_class) in \
                    enumerate(zip(mpole_expn_classes, local_expn_classes)):
                tctx = t.ToyContext(
                    ctx,
                    knl,
                    extra_kernel_kwargs=extra_kernel_kwargs,
                    local_expn_class=local_expn_class,
                    mpole_expn_class=mpole_expn_class,
                )
                pt_src = t.PointSources(
                    tctx,
                    sources,
                    np.ones(sources.shape[-1])
                )

                mexp = t.multipole_expand(
                    pt_src,
                    center=mpole_center.reshape(dim),
                    order=order,
                    rscale=h/order)
                mexp2 = t.multipole_expand(
                    mexp,
                    center=second_center.reshape(dim),
                    order=order,
                    rscale=h/order)
                m2m_vals[i] = mexp2.eval(targets)

            err = np.linalg.norm(m2m_vals[1] - m2m_vals[0]) \
                / np.linalg.norm(m2m_vals[1])
            print(err, distances[-1])
            errs.append(err)

        data.append({'order': order, 'h': distances, 'error': errs})
        import json
        name = type(knl).__name__
        with open(f'{name}_{dim}D_p2m2m2p_error.json', 'w') as f:
            json.dump(data, f, indent=2)


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(LaplaceKernel(2))

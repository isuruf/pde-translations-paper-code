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


def generate(knl, assumption=True):
    extra_kernel_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        if assumption:
            extra_kernel_kwargs = {'k': 1}
        else:
            extra_kernel_kwargs = {'k': 50}

    dim = knl.dim

    mpole_expn_classes = [LinearPDEConformingVolumeTaylorMultipoleExpansion,
                          VolumeTaylorMultipoleExpansion]
    local_expn_classes = [LinearPDEConformingVolumeTaylorLocalExpansion,
                          VolumeTaylorLocalExpansion]

    eval_center = np.array([1, 1, 1][:dim]).reshape(dim, 1)

    ntargets_per_dim = 50
    nsources_per_dim = 50

    sources_grid = np.meshgrid(*[np.linspace(0, 1, nsources_per_dim)
                                 for _ in range(dim)])
    sources_grid = np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1)

    targets_grid = np.meshgrid(*[np.linspace(0, 1, ntargets_per_dim)
                                 for _ in range(dim)])
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1)

    targets = eval_center - (targets_grid - 0.5)

    np.random.seed(1)
    weights = np.random.rand(sources_grid.shape[-1])

    ctx = cl.create_some_context()
    max_order = 12
    if dim == 2 and isinstance(knl, HelmholtzKernel):
        max_order = 12

    if assumption:
        h_values = 2.0**np.arange(-10, -1)
    else:
        h_values = 2.0**np.arange(-10, -3)

    data = []
    direct_vals = [None for _ in h_values]

    for order in range(2, max_order + 1, 2):
        print(order)
        r2s = []
        errs = []
        errs_uncompressed = []
        errs_compressed = []
        data.append({
            'order': order,
            'r2': r2s,
            'r': list(h_values),
            'error': errs,
            'error_uncompressed': errs_uncompressed,
            'error_compressed': errs_compressed,
            'k': extra_kernel_kwargs.get('k', 0),
        })
        for ih, h in enumerate(h_values):
            mpole_center = np.array([h, h, h][:dim]).reshape(dim, 1)
            sources = (2*h*(-0.5+sources_grid.astype(np.float64)) + mpole_center)
            second_center = mpole_center - h
            r1 = np.max(np.linalg.norm(sources - mpole_center, axis=0))
            r2 = r1 + np.linalg.norm(second_center - mpole_center)
            r2s.append(r2)
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
                    weights,
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
                if not assumption:
                    if direct_vals[ih] is None:
                        direct = pt_src.eval(targets)
                        direct_vals[ih] = direct
                    else:
                        direct = direct_vals[ih]

            err = np.linalg.norm(m2m_vals[1] - m2m_vals[0]) \
                / np.linalg.norm(m2m_vals[1])

            if not assumption:
                err_uncompressed = np.linalg.norm(m2m_vals[1] - direct) \
                    / np.linalg.norm(direct)
                err_compressed = np.linalg.norm(m2m_vals[0] - direct) \
                    / np.linalg.norm(direct)
                errs_uncompressed.append(err_uncompressed)
                errs_compressed.append(err_compressed)
                print(r2, err, err_uncompressed, err_compressed)
            else:
                print(r2, err)

            errs.append(err)

        import json
        name = type(knl).__name__
        if assumption:
            fname = f'{name}_{dim}D_p2m2m2p_error.json'
        else:
            fname = f'{name}_{dim}D_p2m2m2p_error_no_assumption.json'

        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(LaplaceKernel(2))

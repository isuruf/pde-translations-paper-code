"""
Compute M2M translation error vs wave number k (fixed source box with side length 2R).
"""

import pyopencl as cl
from arraycontext import PyOpenCLArrayContext
import sumpy.toys as t
import numpy as np
from sumpy.kernel import HelmholtzKernel
import sys

from sumpy.expansion.multipole import (
        VolumeTaylorMultipoleExpansion,
        LinearPDEConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion,
        LinearPDEConformingVolumeTaylorLocalExpansion)


def generate(knl, R=1e-2, k_values=None):
    if k_values is None:
        k_values = np.logspace(0, np.log10(50), 10)

    dim = knl.dim

    mpole_expn_classes = [LinearPDEConformingVolumeTaylorMultipoleExpansion,
                          VolumeTaylorMultipoleExpansion]
    local_expn_classes = [LinearPDEConformingVolumeTaylorLocalExpansion,
                          VolumeTaylorLocalExpansion]

    eval_center = np.array([1]*dim).reshape(dim, 1)

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
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)
    max_order = 12

    # Fixed source box with side length 2R
    mpole_center = np.array([R]*dim).reshape(dim, 1)
    sources = (2*R*(-0.5+sources_grid.astype(np.float64)) + mpole_center)
    second_center = mpole_center - R

    data = []
    direct_vals = [None for _ in k_values]

    for order in range(2, max_order + 1, 2):
        print(order)
        rel_errs = []
        trunc_errs_uncompressed = []
        trunc_errs_compressed = []
        data.append({
            'order': order,
            'R': R,
            'k': list(k_values),
            'rel_error': rel_errs,
            'trunc_error_uncompressed': trunc_errs_uncompressed,
            'trunc_error_compressed': trunc_errs_compressed,
        })
        for ik, k in enumerate(k_values):
            extra_kernel_kwargs = {'k': float(k)}
            m2m_vals = [0, 0]
            for i, (mpole_expn_class, local_expn_class) in \
                    enumerate(zip(mpole_expn_classes, local_expn_classes)):
                tctx = t.ToyContext(
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
                    actx,
                    pt_src,
                    center=mpole_center.reshape(dim),
                    order=order,
                    rscale=R/order)
                mexp2 = t.multipole_expand(
                    actx,
                    mexp,
                    center=second_center.reshape(dim),
                    order=order,
                    rscale=R/order)
                m2m_vals[i] = mexp2.eval(actx, targets)
                if direct_vals[ik] is None:
                    direct = pt_src.eval(actx, targets)
                    direct_vals[ik] = direct
                else:
                    direct = direct_vals[ik]

            # relative error between compressed and uncompressed m2m translation
            rel_err = np.linalg.norm(m2m_vals[1] - m2m_vals[0]) \
                / np.linalg.norm(m2m_vals[1])

            # truncation error for uncompressed/compressed m2m translation
            trunc_err_uncompressed = np.linalg.norm(m2m_vals[1] - direct) \
                / np.linalg.norm(direct)
            trunc_err_compressed = np.linalg.norm(m2m_vals[0] - direct) \
                / np.linalg.norm(direct)
            trunc_errs_uncompressed.append(trunc_err_uncompressed)
            trunc_errs_compressed.append(trunc_err_compressed)
            print(f"  k={k:8.3f}  rel_err={rel_err:.4e}  trunc_err_full={trunc_err_uncompressed:.4e}  trunc_err_cmp={trunc_err_compressed:.4e}")

            rel_errs.append(rel_err)

        import json
        name = type(knl).__name__
        fname = f'{name}_{dim}D_p2m2m2p_error_vs_k_R.json'

        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(HelmholtzKernel(2))

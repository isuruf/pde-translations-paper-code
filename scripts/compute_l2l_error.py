import pyopencl as cl
from arraycontext import PyOpenCLArrayContext
import sumpy.toys as t
import numpy as np
from sumpy.kernel import (HelmholtzKernel, LaplaceKernel,  # noqa: F401
                          BiharmonicKernel)
import sys

from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion,
        LinearPDEConformingVolumeTaylorLocalExpansion)


def generate(knl):
    extra_kernel_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kernel_kwargs = {'k': 1}

    dim = knl.dim

    local_expn_classes = [LinearPDEConformingVolumeTaylorLocalExpansion,
                          VolumeTaylorLocalExpansion]

    source_center = np.array([1]*dim).reshape(dim, 1)

    ntargets_per_dim = 50
    nsources_per_dim = 50

    sources_grid = np.meshgrid(*[np.linspace(0, 1, nsources_per_dim)
                                 for _ in range(dim)])
    sources_grid = np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1)

    targets_grid = np.meshgrid(*[np.linspace(0, 1, ntargets_per_dim)
                                 for _ in range(dim)])
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1)

    sources = source_center - (sources_grid - 0.5)

    np.random.seed(1)
    weights = np.random.rand(sources_grid.shape[-1])

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)
    max_order = 12

    h_values = 2.0**np.arange(-10, -2)

    data = []

    for order in range(2, max_order + 1, 2):
        print(order)
        errs = []
        data.append({
            'order': order,
            'r': list(h_values),
            'error': errs,
            'k': extra_kernel_kwargs.get('k', 0),
        })
        for h in h_values:
            first_local_center = np.zeros((dim, 1))
            second_local_center = first_local_center + h
            targets = (2*h*(-0.5+targets_grid.astype(np.float64)) + second_local_center)
            l2l_vals = [0, 0]
            for i, local_expn_class in enumerate(local_expn_classes):
                tctx = t.ToyContext(
                    knl,
                    extra_kernel_kwargs=extra_kernel_kwargs,
                    local_expn_class=local_expn_class,
                )
                pt_src = t.PointSources(
                    tctx,
                    sources,
                    weights,
                )

                lexp = t.local_expand(
                    actx,
                    pt_src,
                    center=first_local_center.reshape(dim),
                    order=order,
                    rscale=h/order)
                lexp2 = t.local_expand(
                    actx,
                    lexp,
                    center=second_local_center.reshape(dim),
                    order=order,
                    rscale=h/order)
                l2l_vals[i] = lexp2.eval(actx, targets)

            err = np.linalg.norm(l2l_vals[1] - l2l_vals[0]) \
                / np.linalg.norm(l2l_vals[1])
            print(h, err)

            errs.append(err)

        import json
        name = type(knl).__name__
        fname = f'{name}_{dim}D_p2l2l2p_error.json'

        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)


# You can run this using
# $ python compute_l2l_error.py 'generate(LaplaceKernel(2))'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(LaplaceKernel(2))

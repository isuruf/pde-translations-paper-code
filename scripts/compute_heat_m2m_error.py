import json

import numpy as np
import pyopencl as cl

from arraycontext import PyOpenCLArrayContext
import sumpy.toys as toys
from sumpy.kernel import HeatKernel


ALPHA = 1.0
ORDERS = range(2, 10)
H_VALUES = 2.0 ** np.arange(-1, -13, -1)

SOURCE_CENTER = np.array([1.0, 0.5])

TARGET_CENTER = np.array([0.0, 2.5])
TARGET_HX = 1.0
TARGET_HT = 0.5

T_SEP = 1.0
L = np.sqrt(4 * ALPHA * T_SEP)


def make_sources(h_x, h_t, npts=11):
    grid = np.linspace(-1, 1, npts)
    xs, ts = np.meshgrid(grid, grid, indexing="xy")
    sources = np.vstack([
        SOURCE_CENTER[0] + h_x * xs.ravel(),
        SOURCE_CENTER[1] + h_t * ts.ravel(),
    ])
    strengths = np.ones(sources.shape[1])
    return sources, strengths


def make_targets(npts=5):
    grid = np.linspace(-1.0, 1.0, npts)
    xs, ts = np.meshgrid(grid, grid, indexing="xy")
    return np.vstack([
        TARGET_CENTER[0] + TARGET_HX * xs.ravel(),
        TARGET_CENTER[1] + TARGET_HT * ts.ravel(),
    ])


def measure_errors(actx, order):
    targets = make_targets()
    toy_ctx = toys.ToyContext(HeatKernel(1),
                              extra_kernel_kwargs={"alpha": ALPHA})

    rscale = 1.0 / order

    errors = []
    for h in H_VALUES:
        h_x = h * L
        h_t = h * T_SEP
        sources, strengths = make_sources(h_x, h_t)

        c5 = SOURCE_CENTER + np.array([-h_x, h_t])

        pt_src = toys.PointSources(toy_ctx, sources, weights=strengths)

        p2m = toys.multipole_expand(actx, pt_src, SOURCE_CENTER,
                                     order=order, rscale=rscale)
        m2m = toys.multipole_expand(actx, p2m, c5,
                                     order=order, rscale=rscale)
        p2m_direct = toys.multipole_expand(actx, pt_src, c5,
                                            order=order, rscale=rscale)

        m2m_vals = np.asarray(m2m.eval(actx, targets)).ravel()
        p2m_direct_vals = np.asarray(p2m_direct.eval(actx, targets)).ravel()
        err = np.linalg.norm(m2m_vals - p2m_direct_vals) \
            / np.linalg.norm(p2m_direct_vals)
        errors.append(err)
    return errors


def main():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    all_data = []
    for order in ORDERS:
        print(order)
        errs = measure_errors(actx, order)
        all_data.append({
            "order": order,
            "h": [float(h) for h in H_VALUES],
            "error": [float(e) for e in errs],
        })

    with open("HeatKernel_1D_m2m_error.json", "w") as f:
        json.dump(all_data, f, indent=2)


if __name__ == "__main__":
    main()

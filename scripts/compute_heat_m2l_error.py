import json

import numpy as np
import pyopencl as cl

from arraycontext import PyOpenCLArrayContext
import sumpy.toys as toys
from sumpy.kernel import HeatKernel


ALPHA = 1.0
ORDERS = range(2, 10)
H_VALUES = 2.0 ** np.arange(-1, -13, -1)

SOURCE_CENTER = np.array([0.0, 0.5])
SOURCE_HX = 1.0
SOURCE_HT = 0.5

TARGET_CENTER = np.array([0.0, 2.5])

T_SEP = 2 * SOURCE_HT
L = np.sqrt(4 * ALPHA * T_SEP)


def make_sources(rng, npts=11):
    grid = np.linspace(-1, 1, npts)
    xs, ts = np.meshgrid(grid, grid, indexing="xy")
    sources = np.vstack([
        SOURCE_HX * xs.ravel() + SOURCE_CENTER[0],
        SOURCE_HT * ts.ravel() + SOURCE_CENTER[1],
    ])
    strengths = rng.random(sources.shape[1])
    return sources, strengths


def make_targets(center, h_x, h_t, npts=5):
    grid = np.linspace(-1.0, 1.0, npts)
    xs, ts = np.meshgrid(grid, grid, indexing="xy")
    return np.vstack([
        center[0] + h_x * xs.ravel(),
        center[1] + h_t * ts.ravel(),
    ])


def measure_errors(actx, order):
    rng = np.random.default_rng(0)
    sources, strengths = make_sources(rng)
    toy_ctx = toys.ToyContext(HeatKernel(1), extra_kernel_kwargs={"alpha": ALPHA})
    pt_src = toys.PointSources(toy_ctx, sources, weights=strengths)

    rscale = 1 / order
    p2m = toys.multipole_expand(actx, pt_src, SOURCE_CENTER,
                                 order=order, rscale=rscale)

    errors = []
    for h in H_VALUES:
        h_x = h * L
        h_t = h * T_SEP
        c2 = TARGET_CENTER - np.array([h_x, h_t])

        m2l = toys.local_expand(actx, p2m, c2, order=order, rscale=rscale)

        targets = make_targets(TARGET_CENTER, h_x, h_t)
        m2l_vals = np.asarray(m2l.eval(actx, targets)).ravel()
        p2m_vals = np.asarray(p2m.eval(actx, targets)).ravel()
        err = np.linalg.norm(m2l_vals - p2m_vals) / np.linalg.norm(p2m_vals)
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

    with open("HeatKernel_1D_m2l_error.json", "w") as f:
        json.dump(all_data, f, indent=2)


if __name__ == "__main__":
    main()

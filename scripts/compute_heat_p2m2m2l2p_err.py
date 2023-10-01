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

from sumpy.expansion.multipole import (
        VolumeTaylorMultipoleExpansion,
        LinearPDEConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion,
        LinearPDEConformingVolumeTaylorLocalExpansion)
from sumpy.expansion.m2l import (NonFFTM2LTranslationClassFactory,
        FFTM2LTranslationClassFactory)
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
    #mpole_expn_class = VolumeTaylorMultipoleExpansion
    #local_expn_class = VolumeTaylorLocalExpansion
    
    extra_kwargs = {}
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
    if isinstance(knl, HeatKernel):
        extra_kwargs["alpha"] = 0.1

    actx = _acf()
    target_kernels = [knl]
    data = []

    origin = np.array([0, 0, 1, 2][-knl.dim:], np.float64)
    ntargets_per_dim = 4
    nsources_per_dim = 3

    sources_grid = np.meshgrid(*[np.linspace(0, 1, nsources_per_dim)
                                 for _ in range(dim)])
    sources_grid = np.ndarray.flatten(np.array(sources_grid)).reshape(dim, -1)
    #sources = actx.from_numpy((-0.5 + sources_grid) + origin[:, np.newaxis])
    nsources = nsources_per_dim**dim

    strengths = actx.from_numpy(np.ones(nsources, dtype=np.float64) * (1/nsources))

    targets_grid = np.meshgrid(*[np.linspace(0, 1, ntargets_per_dim)
                                 for _ in range(dim)])
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1)
    #targets = eval_offset[:, np.newaxis] + 0.25 * (targets_grid - 0.5)
    #targets = actx.from_numpy(make_obj_array(list(targets)))
    ntargets = ntargets_per_dim**dim

    if dim == 2:
        orders = list(range(2, 25, 2))
    else:
        orders = list(range(2, 13, 2))

    nboxes = 4

    def eval_at(e2p, source_box_nr, rscale, centers):
        e2p_target_boxes = actx.from_numpy(
            np.array([source_box_nr], dtype=np.int32))

        # These are indexed by global box numbers.
        e2p_box_target_starts = actx.from_numpy(
            np.array([0, 0, 0, 0], dtype=np.int32))
        e2p_box_target_counts_nonchild = actx.from_numpy(
            np.array([0, 0, 0, 0], dtype=np.int32))
        e2p_box_target_counts_nonchild[source_box_nr] = ntargets

        evt, (pot,) = e2p(
                actx.queue,

                src_expansions=mpoles,
                src_base_ibox=0,

                target_boxes=e2p_target_boxes,
                box_target_starts=e2p_box_target_starts,
                box_target_counts_nonchild=e2p_box_target_counts_nonchild,
                centers=centers,
                targets=targets,

                rscale=rscale,
                **extra_kwargs
                )
        pot = actx.to_numpy(pot)

        return pot

    m2l_factory = NonFFTM2LTranslationClassFactory()
    m2l_translation = m2l_factory.get_m2l_translation_class(knl, local_expn_class)()

    order = 6
    #for order in orders:
    for h in [1/2**i for i in range(15)]:
        results = {"h": h, "order": order}
        sources = actx.from_numpy(h*(-0.5 + sources_grid) + origin[:, np.newaxis])
        eval_offset = np.array([0.0, 0.0, 0.0, 9.5][-knl.dim:])

        targets = h * (eval_offset[:, np.newaxis] + 0.25 * (targets_grid - 0.5))
        targets = actx.from_numpy(make_obj_array(list(targets)))
        centers = actx.from_numpy((np.array(
            [
                # box 0: particles, first mpole here
                [0, 0, 0, 0][-knl.dim:],

                # box 1: second mpole here
                np.array([-0.1*h, 0, 0, 0][:knl.dim], np.float64),

                # box 2: first local here
                h*(eval_offset + np.array([0, 0.1, 0.2, 0.5][-knl.dim:], np.float64)),

                # box 3: second local and eval here
                h*eval_offset
                ],
            dtype=np.float64) + origin).T.copy())

        m_expn = mpole_expn_class(knl, order=order)
        l_expn = local_expn_class(knl, order=order, m2l_translation=m2l_translation)

        from sumpy import P2EFromSingleBox, E2PFromSingleBox, P2P, E2EFromCSR
        p2m = P2EFromSingleBox(actx.context, m_expn)
        m2m = E2EFromCSR(actx.context, m_expn, m_expn)
        m2p = E2PFromSingleBox(actx.context, m_expn, target_kernels)
        m2l = E2EFromCSR(actx.context, m_expn, l_expn)
        l2l = E2EFromCSR(actx.context, l_expn, l_expn)
        l2p = E2PFromSingleBox(actx.context, l_expn, target_kernels)
        p2p = P2P(actx.context, target_kernels, exclude_self=False)
        # {{{ compute (direct) reference solution

        evt, (pot_direct,) = p2p(
                actx.queue,
                targets, sources, (strengths,),
                **extra_kwargs)
        pot_direct = actx.to_numpy(pot_direct)

        # }}}

        rscale_mult = h
        m1_rscale = 0.5 * rscale_mult
        m2_rscale = 0.25  * rscale_mult
        l1_rscale = 0.5 * rscale_mult
        l2_rscale = 0.25 * rscale_mult

        # {{{ apply P2M

        p2m_source_boxes = actx.from_numpy(np.array([0], dtype=np.int32))

        # These are indexed by global box numbers.
        p2m_box_source_starts = actx.from_numpy(
            np.array([0, 0, 0, 0], dtype=np.int32))
        p2m_box_source_counts_nonchild = actx.from_numpy(
            np.array([nsources, 0, 0, 0], dtype=np.int32))

        evt, (mpoles,) = p2m(actx.queue,
                source_boxes=p2m_source_boxes,
                box_source_starts=p2m_box_source_starts,
                box_source_counts_nonchild=p2m_box_source_counts_nonchild,
                centers=centers,
                sources=sources,
                strengths=(strengths,),
                nboxes=nboxes,
                rscale=m1_rscale,

                tgt_base_ibox=0,
                **extra_kwargs)

        # }}}

        pot_prev = pot_direct
        pot = eval_at(m2p, 0, m1_rscale, centers)
        err = la.norm(pot - pot_prev) / la.norm(pot_prev)
        pot_prev = pot
        results["p2m2p/p2p"] = err
        #print(err)

        # {{{ apply M2M

        m2m_target_boxes = actx.from_numpy(np.array([1], dtype=np.int32))
        m2m_src_box_starts = actx.from_numpy(np.array([0, 1], dtype=np.int32))
        m2m_src_box_lists = actx.from_numpy(np.array([0], dtype=np.int32))

        evt, (mpoles,) = m2m(actx.queue,
                src_expansions=mpoles,
                src_base_ibox=0,
                tgt_base_ibox=0,
                ntgt_level_boxes=mpoles.shape[0],

                target_boxes=m2m_target_boxes,

                src_box_starts=m2m_src_box_starts,
                src_box_lists=m2m_src_box_lists,
                centers=centers,

                src_rscale=m1_rscale,
                tgt_rscale=m2_rscale,
                **extra_kwargs)

        # }}}

        pot = eval_at(m2p, 1, m2_rscale, centers)
        err = la.norm(pot - pot_prev) / la.norm(pot_prev)
        pot_prev = pot
        results["p2m2m2p/p2m2p"] = err
        #print(err)

        # {{{ apply M2L

        m2l_target_boxes = actx.from_numpy(np.array([2], dtype=np.int32))
        m2l_src_box_starts = actx.from_numpy(np.array([0, 1], dtype=np.int32))
        m2l_src_box_lists = actx.from_numpy(np.array([1], dtype=np.int32))

        evt, (mpoles,) = m2l(actx.queue,
                src_expansions=mpoles,
                src_base_ibox=0,
                tgt_base_ibox=0,
                ntgt_level_boxes=mpoles.shape[0],

                target_boxes=m2l_target_boxes,
                src_box_starts=m2l_src_box_starts,
                src_box_lists=m2l_src_box_lists,
                centers=centers,

                src_rscale=m2_rscale,
                tgt_rscale=l1_rscale,
                **extra_kwargs)

        # }}}

        pot = eval_at(l2p, 2, l1_rscale, centers)
        err = la.norm(pot - pot_prev) / la.norm(pot_prev)
        pot_prev = pot
        results["p2m2m2l2p/p2m2m2p"] = err
        #print(err)

        # {{{ apply L2L

        l2l_target_boxes = actx.from_numpy(np.array([3], dtype=np.int32))
        l2l_src_box_starts = actx.from_numpy(np.array([0, 1], dtype=np.int32))
        l2l_src_box_lists = actx.from_numpy(np.array([2], dtype=np.int32))

        evt, (mpoles,) = l2l(actx.queue,
                src_expansions=mpoles,
                src_base_ibox=0,
                tgt_base_ibox=0,
                ntgt_level_boxes=mpoles.shape[0],

                target_boxes=l2l_target_boxes,
                src_box_starts=l2l_src_box_starts,
                src_box_lists=l2l_src_box_lists,
                centers=centers,

                src_rscale=l1_rscale,
                tgt_rscale=l2_rscale,
                **extra_kwargs)

        # }}}

        pot = eval_at(l2p, 3, l2_rscale, centers)

        err = la.norm(pot - pot_direct) / la.norm(pot_direct)
        results["p2m2m2l2l2p/p2p"] = err
        data.append(results)
        print(results)
        
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

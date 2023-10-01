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
    sources = actx.from_numpy((-0.5 + sources_grid) + origin[:, np.newaxis])
    nsources = nsources_per_dim**dim

    strengths = actx.from_numpy(np.ones(nsources, dtype=np.float64) * (1/nsources))

    targets_grid = np.meshgrid(*[np.linspace(0, 1, ntargets_per_dim)
                                 for _ in range(dim)])
    targets_grid = np.ndarray.flatten(np.array(targets_grid)).reshape(dim, -1)
    targets = eval_offset[:, np.newaxis] + 0.25 * (targets_grid - 0.5)
    targets = actx.from_numpy(make_obj_array(list(targets)))
    ntargets = ntargets_per_dim**dim

    centers = actx.from_numpy((np.array(
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
            dtype=np.float64) + origin).T.copy())

    del eval_offset

    if dim == 2:
        orders = list(range(2, 25, 2))
    else:
        orders = list(range(2, 13, 2))

    nboxes = centers.shape[-1]

    def eval_at(e2p, source_box_nr, rscale):
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

    m2l_factory = FFTM2LTranslationClassFactory()
    m2l_translation = m2l_factory.get_m2l_translation_class(knl, local_expn_class)()
    print(m2l_translation)

    for order in orders:
        m_expn = mpole_expn_class(knl, order=order)
        l_expn = local_expn_class(knl, order=order, m2l_translation=m2l_translation)

        from sumpy import (P2EFromSingleBox, E2PFromSingleBox, P2P, E2EFromCSR,
            M2LUsingTranslationClassesDependentData,
            M2LGenerateTranslationClassesDependentData,
            M2LPreprocessMultipole, M2LPostprocessLocal)

        p2m = P2EFromSingleBox(actx.context, m_expn)
        m2m = E2EFromCSR(actx.context, m_expn, m_expn)
        m2p = E2PFromSingleBox(actx.context, m_expn, target_kernels)
        m2l_data = M2LGenerateTranslationClassesDependentData(actx.context, m_expn, l_expn)
        m2l_pre = M2LPreprocessMultipole(actx.context, m_expn, l_expn)
        m2l = M2LUsingTranslationClassesDependentData(actx.context, m_expn, l_expn)
        m2l_post = M2LPostprocessLocal(actx.context, m_expn, l_expn)
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

        m1_rscale = 0.5
        m2_rscale = 0.25
        l1_rscale = 0.5
        l2_rscale = 0.25

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
                wait_for=[evt],
                **extra_kwargs)

        # }}}

        pot = eval_at(m2p, 0, m1_rscale)
        err = la.norm(pot - pot_direct) / la.norm(pot_direct)
        print(err)

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
                wait_for=[evt],
                **extra_kwargs)

        # }}}

        pot = eval_at(m2p, 1, m2_rscale)
        err = la.norm(pot - pot_direct) / la.norm(pot_direct)
        print(err)

        # {{{ apply M2L
        
        print(actx.to_numpy(mpoles[1:2, :]))
        m2l_target_boxes = actx.from_numpy(np.array([0], dtype=np.int32))
        m2l_src_box_starts = actx.from_numpy(np.array([0, 1], dtype=np.int32))
        m2l_src_box_lists = actx.from_numpy(np.array([0], dtype=np.int32))

        len_fft = m2l_translation.preprocess_multipole_nexprs(l_expn, m_expn)
        m2l_preprocess = actx.zeros((1, len_fft), dtype=np.complex128)

        # Preprocess the mpole expansion at box 1
        evt, _ = m2l_pre(actx.queue,
                src_expansions=mpoles[1:2, :],
                preprocessed_src_expansions=m2l_preprocess,
                src_rscale=np.float64(m2_rscale),
                wait_for=[evt],
                **extra_kwargs)

        from sumpy.tools import run_opencl_fft, get_opencl_fft_app, get_native_event
        fft_app = get_opencl_fft_app(actx.queue, (len_fft,),
            dtype=m2l_preprocess.dtype, inverse=False)
        ifft_app = get_opencl_fft_app(actx.queue, (len_fft,),
            dtype=m2l_preprocess.dtype, inverse=True)

        evt, m2l_preprocess = run_opencl_fft(fft_app, actx.queue,
                m2l_preprocess, inverse=False, wait_for=[evt])

        m2l_translation_classes_lists = actx.from_numpy(np.array([0], dtype=np.int32))
        centers_np = actx.to_numpy(centers)
        dist = centers_np[:, 2] - centers_np[:, 1]
        m2l_translation_vectors = actx.from_numpy(dist.reshape(dim, 1))
        m2l_translation_classes_dependent_data = actx.zeros(
                (1, len_fft), dtype=np.complex128)

        # translation classes data
        evt, _ = m2l_data(
                actx.queue,
                src_rscale=m2_rscale,
                ntranslation_classes=1,
                translation_classes_level_start=0,
                m2l_translation_classes_dependent_data=(
                    m2l_translation_classes_dependent_data),
                m2l_translation_vectors=m2l_translation_vectors,
                ntranslation_vectors=1,
                wait_for=[get_native_event(evt)],
                **extra_kwargs)

        evt, m2l_translation_classes_dependent_data = run_opencl_fft(fft_app, actx.queue,
                m2l_translation_classes_dependent_data, inverse=False, wait_for=[evt])

        print(m2l_preprocess.shape, mpoles.shape)
        # translate part
        evt, (local_before,) = m2l(actx.queue,
                src_expansions=m2l_preprocess,
                src_base_ibox=0,
                tgt_base_ibox=0,
                ntgt_level_boxes=1,

                target_boxes=m2l_target_boxes,
                src_box_starts=m2l_src_box_starts,
                src_box_lists=m2l_src_box_lists,
                centers=centers,

                src_rscale=m2_rscale,
                tgt_rscale=l1_rscale,
                m2l_translation_classes_dependent_data=(
                    m2l_translation_classes_dependent_data),
                m2l_translation_classes_lists=m2l_translation_classes_lists,
                translation_classes_level_start=0,
                wait_for=[get_native_event(evt)],
                **extra_kwargs)
        
        print(local_before.shape)

        # iFFT
        evt, local_before = run_opencl_fft(ifft_app, actx.queue,
            local_before, inverse=True, wait_for=[evt])

        # Postprocess the local expansion
        evt, _ = m2l_post(queue=actx.queue,
                tgt_expansions_before_postprocessing=local_before,
                tgt_expansions=mpoles[2:3, :],
                src_rscale=np.float64(m2_rscale),
                tgt_rscale=np.float64(l1_rscale),
                wait_for=[get_native_event(evt)],
                **extra_kwargs)
        # }}}

        pot = eval_at(l2p, 2, l1_rscale)
        err = la.norm(pot - pot_direct) / la.norm(pot_direct)
        print(err)

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

        pot = eval_at(l2p, 3, l2_rscale)
        err = la.norm(pot - pot_direct) / la.norm(pot_direct)
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

"""
Run this script using gitlab.tiker.net/isuruf/sumpy branch m2m
Also need https://gitlab.tiker.net/inducer/sumpy/merge_requests/113
"""

from sumpy.kernel import (HelmholtzKernel, LaplaceKernel)  # noqa:F401
import sys

from sumpy.expansion.multipole import (
        VolumeTaylorMultipoleExpansion,
        LinearPDEConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
        VolumeTaylorLocalExpansion,
        LinearPDEConformingVolumeTaylorLocalExpansion)

import pymbolic.mapper.flop_counter
from pymbolic.algorithm import find_factors

import sumpy.symbolic as sym
from sumpy.assignment_collection import SymbolicAssignmentCollection
from sumpy.codegen import to_loopy_insns


def fft_flop_count(n):
    m = n
    count = 0
    while m > 1:
        n1, n2 = find_factors(m)
        count += n1*n
        m = n2
    return count


def generate(knl):

    dim = knl.dim

    mpole_expn_classes = [LinearPDEConformingVolumeTaylorMultipoleExpansion,
                          VolumeTaylorMultipoleExpansion]
    local_expn_classes = [LinearPDEConformingVolumeTaylorLocalExpansion,
                          VolumeTaylorLocalExpansion]

    from collections import defaultdict

    data = {'order': []}
    for op in ['m2l', 'p2m', 'm2m', 'm2p', 'p2l', 'l2p', 'l2l']:
        data[op] = defaultdict(list)
    names = ['compressed_flops', 'full_flops']

    def count_flops(exprs, sac=None):
        if sac is None:
            sac = SymbolicAssignmentCollection()
        for i, expr in enumerate(exprs):
            sac.assign_unique('coeff%d' % i, expr)
        sac.run_global_cse()
        insns = to_loopy_insns(sac.assignments.items())
        counter = pymbolic.mapper.flop_counter.CSEAwareFlopCounter()
        s = 0
        for insn in insns:
            s += counter(insn.expression)
        return s

    dvec = sym.make_sym_vector('d', knl.dim)
    src_rscale = sym.Symbol('src_rscale')
    tgt_rscale = sym.Symbol('tgt_rscale')

    if dim == 2:
        max_order = 40
    else:
        max_order = 40
    
    from sumpy.expansion.m2l import (
                VolumeTaylorM2LWithFFT, VolumeTaylorM2LTranslation)

    fft = True
    if fft:
        m2l_translation = VolumeTaylorM2LWithFFT()
    else:
        m2l_translation = VolumeTaylorM2LTranslation()

    for order in range(2, max_order, 2):
        print(order)
        m2l_flops = []
        for i, (mpole_expn_class, local_expn_class) in \
                enumerate(zip(mpole_expn_classes, local_expn_classes)):
            m_expn = mpole_expn_class(knl, order=order)
            l_expn = local_expn_class(
                kernel=knl, order=order,
                m2l_translation=m2l_translation)

            src_coeff_exprs = [
                sym.Symbol(f'src_coeff{i}') for i in range(len(m_expn))]

            # { {{ M2L
            m2l_translation_classes_dependent_data = [
                sym.Symbol(f'data{i}') for i in range(len(m_expn))]
            m2l_result = [
                sym.Symbol(f'result{i}') for i in range(
                    l_expn.m2l_translation.postprocess_local_nexprs(l_expn, m_expn))]

            sac = SymbolicAssignmentCollection()
            result = l_expn.translate_from(
                m_expn, src_coeff_exprs,
                src_rscale=src_rscale,
                dvec=dvec, tgt_rscale=tgt_rscale, sac=sac,
                m2l_translation_classes_dependent_data=(
                    m2l_translation_classes_dependent_data))
            m2l_flops = count_flops(result, sac=sac)
            # sac = SymbolicAssignmentCollection()
            # result = l_expn.m2l_postprocess_local_exprs(m_expn, m2l_result,
            #     src_rscale, tgt_rscale, sac=sac)
            # f = count_flops(result, sac=sac)
            f2 = fft_flop_count(len(m2l_result))
            m2l_flops += 2*f2 / (6**dim - 3**dim)
            # }}}

            sac = SymbolicAssignmentCollection()
            result = m_expn.coefficients_from_source(
                knl, dvec, None, rscale=src_rscale, sac=sac)
            p2m_flops = count_flops(result, sac=sac)

            sac = SymbolicAssignmentCollection()
            result = l_expn.coefficients_from_source(
                knl, dvec, None, rscale=src_rscale, sac=sac)
            p2l_flops = count_flops(result, sac=sac)

            sac = SymbolicAssignmentCollection()
            result = m_expn.translate_from(
                m_expn, src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac=sac)
            m2m_flops = count_flops(result, sac=sac)

            sac = SymbolicAssignmentCollection()
            result = m_expn.evaluate(
                knl, src_coeff_exprs, dvec, src_rscale, sac=sac)
            m2p_flops = count_flops([result], sac=sac)

            sac = SymbolicAssignmentCollection()
            result = l_expn.evaluate(
                knl, src_coeff_exprs, dvec, src_rscale, sac=sac)
            l2p_flops = count_flops([result], sac=sac)

            sac = SymbolicAssignmentCollection()
            result = l_expn.translate_from(
                l_expn, src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac=sac)
            l2l_flops = count_flops(result, sac=sac)

            data['m2m'][names[i]].append(m2m_flops)
            data['m2p'][names[i]].append(m2p_flops)
            data['m2l'][names[i]].append(m2l_flops)
            data['p2m'][names[i]].append(p2m_flops)
            data['p2l'][names[i]].append(p2l_flops)
            data['l2p'][names[i]].append(l2p_flops)
            data['l2l'][names[i]].append(l2l_flops)
        data['order'].append(order)
        print(data)
        import json
        with open('{}_{}D_flop_count.json'.format(type(knl).__name__, dim),
                  'w') as f:
            json.dump(data, f, indent=2)


# You can run this using
# $ python generate_data.py 'generate(LaplaceKernel(2))'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        generate(LaplaceKernel(2))

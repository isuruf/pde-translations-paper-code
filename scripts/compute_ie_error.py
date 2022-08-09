from __future__ import division, absolute_import, print_function

import numpy as np
import numpy.linalg as la
from pytools import RecordWithoutPickling, memoize_method
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from functools import partial
from meshmode.mesh.generation import ellipse, make_curve_mesh
from meshmode.discretization.visualization import make_visualizer
from meshmode.dof_array import flatten_to_numpy
import meshmode
from sumpy.kernel import LaplaceKernel, HelmholtzKernel, BiharmonicKernel

from pytential import bind, sym
from pytential import GeometryCollection
from pytools.obj_array import flat_obj_array
from pytential.qbx import QBXTargetAssociationFailedException

import json
import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)


# {{{ helpers

def make_circular_point_group(
        ambient_dim, npoints, radius,
        center=np.array([0., 0.]), func=lambda x: x
        ):
    t = func(np.linspace(0, 1, npoints, endpoint=False)) * (2 * np.pi)
    center = np.asarray(center)
    result = np.zeros((ambient_dim, npoints))
    result[:2, :] = center[:, np.newaxis] + \
        radius*np.vstack((np.cos(t), np.sin(t)))
    return result


# }}}

# {{{ test cases
class IntegralEquationTestCase(RecordWithoutPickling):
    name = 'unknown'

    # operator
    knl_class_or_helmholtz_k = 0
    knl_kwargs = {}
    bc_type = 'dirichlet'
    side = -1

    # qbx
    qbx_order = None
    source_ovsmp = 4
    target_order = None
    use_refinement = True

    # fmm
    fmm_backend = 'sumpy'
    fmm_order = None
    fmm_tol = None

    # solver
    gmres_tol = 1e-9

    # test case
    resolutions = None
    inner_radius = None
    outer_radius = None
    check_tangential_deriv = True
    check_gradient = False

    def __init__(self, **kwargs):
        import inspect
        members = inspect.getmembers(
                type(self), lambda m: not inspect.isroutine(m))
        members = dict(
                m for m in members
                if (not m[0].startswith('__')
                    and m[0] != 'fields'
                    and not isinstance(m[1], property))
                )

        for k, v in kwargs.items():
            if k not in members:
                raise KeyError(f'unknown keyword argument "{k}"')
            members[k] = v

        super().__init__(**members)

    # {{{ symbolic

    @property
    @memoize_method
    def knl_class(self):
        if isinstance(self.knl_class_or_helmholtz_k, type):
            return self.knl_class_or_helmholtz_k

        if self.knl_class_or_helmholtz_k == 0:
            from sumpy.kernel import LaplaceKernel
            return LaplaceKernel
        else:
            from sumpy.kernel import HelmholtzKernel
            return HelmholtzKernel

    @property
    @memoize_method
    def knl_concrete_kwargs(self):
        if isinstance(self.knl_class_or_helmholtz_k, type):
            return self.knl_kwargs

        kwargs = self.knl_kwargs.copy()
        if self.knl_class_or_helmholtz_k != 0:
            kwargs['k'] = self.knl_class_or_helmholtz_k

        return kwargs

    @property
    @memoize_method
    def knl_sym_kwargs(self):
        return {k: sym.var(k) for k in self.knl_concrete_kwargs}

    def get_operator(self, ambient_dim):
        sign = +1 if self.side in [+1, 'scat'] else -1
        knl = self.knl_class(ambient_dim)   # noqa: pylint:disable=E1102

        if self.bc_type == 'dirichlet':
            from pytential.symbolic.pde.scalar import DirichletOperator
            op = DirichletOperator(
                    knl, sign,
                    use_l2_weighting=True,
                    kernel_arguments=self.knl_sym_kwargs)
        elif self.bc_type == 'neumann':
            from pytential.symbolic.pde.scalar import NeumannOperator
            op = NeumannOperator(
                    knl, sign,
                    use_l2_weighting=True,
                    use_improved_operator=False,
                    kernel_arguments=self.knl_sym_kwargs)
        elif self.bc_type == 'clamped_plate':
            from pytential.symbolic.pde.scalar import \
                    BiharmonicClampedPlateOperator
            op = BiharmonicClampedPlateOperator(knl, sign)
        else:
            raise ValueError(f'unknown bc_type: "{self.bc_type}"')

        return op

    # }}}

    # {{{ geometry

    def get_mesh(self, resolution, mesh_order):
        raise NotImplementedError

    def get_discretization(self, actx, resolution, mesh_order):
        mesh = self.get_mesh(resolution, mesh_order)

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory
        return Discretization(
                actx, mesh,
                InterpolatoryQuadratureSimplexGroupFactory(self.target_order))

    def get_layer_potential(self, actx, resolution, mesh_order, algorithm):
        pre_density_discr = self.get_discretization(
            actx, resolution, mesh_order)

        from sumpy.expansion.level_to_order import SimpleExpansionOrderFinder
        fmm_kwargs = {}
        if self.fmm_backend is None:
            fmm_kwargs['fmm_order'] = False
        else:
            if self.fmm_tol is not None:
                fmm_kwargs['fmm_order'] = SimpleExpansionOrderFinder(
                        self.fmm_tol)
            elif self.fmm_order is not None:
                fmm_kwargs['fmm_order'] = self.fmm_order
            else:
                fmm_kwargs['fmm_order'] = self.qbx_order + 5

        from sumpy.expansion import (
                VolumeTaylorExpansionFactory, DefaultExpansionFactory)
        from sumpy.expansion.m2l import (
                VolumeTaylorM2LWithFFT, VolumeTaylorM2LTranslation)

        if algorithm == 'compressed_fft':
            m2l_translation = VolumeTaylorM2LWithFFT()
        else:
            m2l_translation = VolumeTaylorM2LTranslation()

        if algorithm == 'full':
            factory_parent_class = VolumeTaylorExpansionFactory
        else:
            factory_parent_class = DefaultExpansionFactory

        class ExpansionFactory(factory_parent_class):
            def get_local_expansion_class(self, base_kernel):
                res = super().get_local_expansion_class(base_kernel)
                return partial(res, m2l_translation=m2l_translation)

            def get_qbx_local_expansion_class(self, base_kernel):
                return partial(super().get_local_expansion_class(base_kernel),
                    m2l_translation=VolumeTaylorM2LTranslation())

        from pytential.qbx import QBXLayerPotentialSource
        return QBXLayerPotentialSource(
                pre_density_discr,
                fine_order=self.source_ovsmp * self.target_order,
                qbx_order=self.qbx_order,
                fmm_backend=self.fmm_backend, **fmm_kwargs,
                expansion_factory=ExpansionFactory(),
                _disable_refinement=not self.use_refinement,
                _box_extent_norm=getattr(self, 'box_extent_norm', None),
                _from_sep_smaller_crit=getattr(
                    self, 'from_sep_smaller_crit', None),
                _from_sep_smaller_min_nsources_cumul=30,
                )

    # }}}

    def __str__(self):
        if not self.__class__.fields:
            return f'{type(self).__name__}()'

        width = len(max(list(self.__class__.fields), key=len))
        fmt = f'%{width}s : %s'

        attrs = {k: getattr(self, k) for k in self.__class__.fields}
        header = {
                'class': type(self).__name__,
                'name': attrs.pop('name'),
                '-' * width: '-' * width
                }

        return '\n'.join([
            '\t%s' % '\n\t'.join(fmt % (k, v) for k, v in header.items()),
            '\t%s' % '\n\t'.join(
                fmt % (k, v) for k, v in sorted(attrs.items())),
            ])


class CurveTestCase(IntegralEquationTestCase):
    ambient_dim = 2

    # qbx
    qbx_order = 5
    target_order = 5

    # fmm
    fmm_backend = None

    # test case
    curve_fn = None
    inner_radius = 0.1
    outer_radius = 2
    resolutions = [40, 50, 60]

    def _curve_fn(self, t):
        return self.curve_fn(t)     # pylint:disable=not-callable

    def get_mesh(self, resolution, mesh_order):
        return make_curve_mesh(
                self._curve_fn,
                np.linspace(0, 1, resolution + 1),
                mesh_order)


class EllipseTestCase(CurveTestCase):
    name = 'ellipse'
    aspect_ratio = 3.0
    radius = 1.0

    def _curve_fn(self, t):
        from meshmode.mesh.generation import ellipse
        return self.radius * ellipse(self.aspect_ratio, t)


def make_source_and_target_points(
        actx, side, inner_radius, outer_radius, ambient_dim,
        nsources=10, ntargets=20):
    if side == -1:
        test_src_geo_radius = outer_radius
        test_tgt_geo_radius = inner_radius
    elif side == +1:
        test_src_geo_radius = inner_radius
        test_tgt_geo_radius = outer_radius
    elif side == 'scat':
        test_src_geo_radius = outer_radius
        test_tgt_geo_radius = outer_radius
    else:
        raise ValueError(f'unknown side: {side}')

    from pytential.source import PointPotentialSource
    point_sources = make_circular_point_group(
            ambient_dim, nsources, test_src_geo_radius)
    point_source = PointPotentialSource(
            actx.freeze(actx.from_numpy(point_sources)))

    from pytential.target import PointsTarget
    test_targets = make_circular_point_group(
            ambient_dim, ntargets, test_tgt_geo_radius)
    point_target = PointsTarget(
        actx.freeze(actx.from_numpy(test_targets)))

    return point_source, point_target

# }}}


def main(algorithm, fmm_order):
    ctx_factory = meshmode._acf
    visualize = True
    actx = ctx_factory()
    qbx_order = 5
    resolution = 100

    case = EllipseTestCase(
        knl_class_or_helmholtz_k=BiharmonicKernel,
        bc_type='clamped_plate', side=-1, fmm_backend='sumpy')
    case.fmm_order = fmm_order
    case.qbx_order = qbx_order

    # {{{ refinement
    refiner_extra_kwargs = {}
    if case.use_refinement:
        if case.knl_class == HelmholtzKernel and \
                getattr(case, 'refine_on_helmholtz_k', True):
            k = case.knl_concrete_kwargs['k']
            refiner_extra_kwargs['kernel_length_scale'] = 5 / k

        if hasattr(case, 'scaled_max_curvature_threshold'):
            refiner_extra_kwargs['scaled_max_curvature_threshold'] = \
                    case.scaled_max_curvature_threshold

        if hasattr(case, 'expansion_disturbance_tolerance'):
            refiner_extra_kwargs['expansion_disturbance_tolerance'] = \
                    case.expansion_disturbance_tolerance

        if hasattr(case, 'refinement_maxiter'):
            refiner_extra_kwargs['maxiter'] = case.refinement_maxiter

        # refiner_extra_kwargs['visualize'] = True
    # }}}

    # {{{ construct geometries

    qbx = case.get_layer_potential(
        actx, resolution, case.target_order, algorithm=algorithm)
    point_source, point_target = make_source_and_target_points(
        actx, case.side, case.inner_radius, case.outer_radius, qbx.ambient_dim)

    places = {
            case.name: qbx,
            'point_source': point_source,
            'point_target': point_target
            }

    # plotting grid points
    ambient_dim = qbx.ambient_dim
    if visualize:
        vis_grid_spacing = getattr(case, 'vis_grid_spacing',
                                   (0.1, 0.1, 0.1)[:ambient_dim])
        vis_extend_factor = getattr(case, 'vis_extend_factor', 0.2)

        from sumpy.visualization import make_field_plotter_from_bbox
        from meshmode.mesh.processing import find_bounding_box
        fplot = make_field_plotter_from_bbox(
                find_bounding_box(qbx.density_discr.mesh),
                h=vis_grid_spacing,
                extend_factor=vis_extend_factor)

        from pytential.target import PointsTarget
        plot_targets = PointsTarget(fplot.points)

        places.update({
            'qbx_target_tol': qbx.copy(target_association_tolerance=0.15),
            'plot_targets': plot_targets
            })

    places = GeometryCollection(places, auto_where=case.name)
    if case.use_refinement:
        from pytential.qbx.refinement import refine_geometry_collection
        places = refine_geometry_collection(places, **refiner_extra_kwargs)

    dd = sym.as_dofdesc(case.name).to_stage1()
    density_discr = places.get_discretization(dd.geometry)

    logger.info('nelements:     %d', density_discr.mesh.nelements)
    logger.info('ndofs:         %d', density_discr.ndofs)

    if case.use_refinement:
        logger.info(
            '%d elements before refinement', qbx.density_discr.mesh.nelements)

        discr = places.get_discretization(
            dd.geometry, sym.QBX_SOURCE_STAGE1)
        logger.info(
            '%d stage-1 elements after refinement', discr.mesh.nelements)

        discr = places.get_discretization(
            dd.geometry, sym.QBX_SOURCE_STAGE2)
        logger.info(
            '%d stage-2 elements after refinement', discr.mesh.nelements)

        discr = places.get_discretization(
            dd.geometry, sym.QBX_SOURCE_QUAD_STAGE2)
        logger.info(
            'quad stage-2 elements have %d nodes', discr.groups[0].nunit_dofs)

    # }}}

    # {{{ set up operator

    knl = case.knl_class(ambient_dim)
    op = case.get_operator(ambient_dim)
    if knl.is_complex_valued:
        dtype = np.complex128
    else:
        dtype = np.float64

    sym_u = op.get_density_var('u')
    sym_bc = op.get_density_var('bc')
    sym_charges = sym.var('charges')

    sym_op_u = op.operator(sym_u)

    # }}}

    # {{{ set up test data

    np.random.seed(22)
    source_charges = np.random.randn(point_source.ndofs)
    source_charges[-1] = -np.sum(source_charges[:-1])
    source_charges = source_charges.astype(dtype)
    assert np.sum(source_charges) < 1.0e-15

    source_charges_dev = actx.from_numpy(source_charges)

    # }}}

    # {{{ establish BCs

    pot_src = sym.int_g_vec(
        # FIXME: qbx_forced_limit--really?
        knl, sym_charges, qbx_forced_limit=None, **case.knl_sym_kwargs)

    test_direct = bind(
        places,
        pot_src,
        auto_where=('point_source', 'point_target'))(
            actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

    if case.bc_type == 'dirichlet':
        bc = bind(
            places,
            pot_src,
            auto_where=('point_source', case.name))(
                actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

    elif case.bc_type == 'neumann':
        bc = bind(
            places,
            sym.normal_derivative(ambient_dim, pot_src, dofdesc=case.name),
            auto_where=('point_source', case.name))(
                actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

    elif case.bc_type == 'clamped_plate':
        bc_u = bind(
            places,
            pot_src,
            auto_where=('point_source', case.name))(
                actx, charges=source_charges_dev, **case.knl_concrete_kwargs)
        bc_du = bind(
            places,
            sym.normal_derivative(ambient_dim, pot_src, dofdesc=case.name),
            auto_where=('point_source', case.name))(
                actx, charges=source_charges_dev, **case.knl_concrete_kwargs)

        bc = flat_obj_array(bc_u, bc_du)
    else:
        raise ValueError(f'unknown bc_type: "{case.bc_type}"')

    # }}}

    # {{{ solve

    bound_op = bind(places, sym_op_u)
    rhs = bind(places, op.prepare_rhs(sym_bc))(actx, bc=bc)

    try:
        from pytential.solve import gmres
        gmres_result = gmres(
            bound_op.scipy_op(actx, 'u', dtype, **case.knl_concrete_kwargs),
            rhs,
            tol=case.gmres_tol,
            progress=True,
            hard_failure=True,
            stall_iterations=100, no_progress_factor=1.05,
            require_monotonicity=False)
    except QBXTargetAssociationFailedException as e:
        bdry_vis = make_visualizer(actx, density_discr, case.target_order + 3)

        bdry_vis.write_vtk_file(f'failed-targets-solve-{resolution}.vtu', [
            ('failed_targets', actx.thaw(e.failed_target_flags)),
            ])
        raise

    logger.info('gmres state: %s', gmres_result.state)
    weighted_u = gmres_result.solution

    # }}}

    # {{{ error computation

    test_via_bdry = bind(
        places,
        op.representation(sym_u),
        auto_where=(case.name, 'point_target')
        )(actx, u=weighted_u, **case.knl_concrete_kwargs)

    err = test_via_bdry - test_direct

    err = flatten_to_numpy(actx, err, strict=False)
    test_direct = flatten_to_numpy(actx, test_direct, strict=False)
    test_via_bdry = flatten_to_numpy(actx, test_via_bdry, strict=False)

    # {{{ remove effect of net source charge

    if (case.knl_class == LaplaceKernel
            and case.bc_type == 'neumann'
            and case.side == -1):
        # remove constant offset in interior Laplace Neumann error
        tgt_ones = np.ones_like(test_direct)
        tgt_ones = tgt_ones / la.norm(tgt_ones)
        err = err - np.vdot(tgt_ones, err) * tgt_ones

    # }}}

    rel_err_2 = la.norm(err) / la.norm(test_direct)
    rel_err_inf = la.norm(err, np.inf) / la.norm(test_direct, np.inf)

    logger.info('rel_err_2: %.5e rel_err_inf: %.5e', rel_err_2, rel_err_inf)

    # }}}

    print(rel_err_inf)
    return rel_err_inf


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fmm_orders = list(range(6, 20, 2))
    all_data = {'fmm_order': fmm_orders}
    for algorithm in ['compressed_fft', 'full', 'compressed']:
        data = []
        all_data[algorithm] = data
        for fmm_order in fmm_orders:
            rel_err_inf = main(algorithm, fmm_order)
            data.append(rel_err_inf)
            with open('Biharmonic_IE_error.json', 'w') as f:
                json.dump(all_data, f)

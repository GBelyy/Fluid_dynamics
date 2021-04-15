
import numpy as np

from pysph.examples._db_geometry import DamBreak3DGeometry

from pysph.base.kernels import WendlandQuintic
from pysph.base.utils import get_particle_array_rigid_body

from pysph.sph.equation import Group
from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
from pysph.sph.wc.basic import TaitEOS, TaitEOSHGCorrection, MomentumEquation

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.tools.gmsh import vtk_file_to_points

from pysph.sph.rigid_body import (
    BodyForce, NumberDensity, RigidBodyForceGPUGems,
    RigidBodyMoments, RigidBodyMotion,
    RK2StepRigidBody, PressureRigidBody
)

dim = 3

dt = 1e-6
tf = 5

dx = 0.5   #volume of the particle
nboundary_layers = 3
hdx = 1   #wall thickness
rho0 = 1000
cloud_height = 6 #start_position of array of fluid

#in-built class of location
class DamBreak3DSPH(Application):
    def initialize(self):
        self.geom = DamBreak3DGeometry(
        container_height=5, container_width=20, container_length=20,
        fluid_column_height=4, fluid_column_width=3, fluid_column_length=3,
        obstacle_center_x=0, obstacle_center_y=0, # parameters object .stl
        #obstacle_length= , obstacle_height= , obstacle_width= ,
        dx=dx, nboundary_layers=nboundary_layers, hdx=hdx, rho0=rho0,
        with_obstacle=False)

        #length - y
        #height - z
        #width - x

    def create_particles(self):
        fluid, boundary = self.geom.create_particles()
        vtl_filename = 'thor.vtk'
        x, y, z = vtk_file_to_points(vtl_filename, cell_centers=False)

        # clarification of .stl position
        difference_x = min(x)
        #difference_y = min(y)
        x -= difference_x
        #y -= difference_y
        x += 7
        z+=1
        fluid.z += cloud_height  # location of fluid column

        m = np.ones_like(x)*fluid.m[0]
        h = np.ones_like(x)*fluid.h[0]
        rho = np.ones_like(x)*fluid.rho[0]

        obstacle = get_particle_array_rigid_body(
            name='obstacle', x=x, y=y, z=z, m=m, h=h, rho=rho, rho0=rho
        )
        obstacle.total_mass[0] = np.sum(m)
        obstacle.add_property('cs')
        obstacle.add_property('arho')
        obstacle.set_lb_props(list(obstacle.properties.keys()))
        obstacle.set_output_arrays(
            ['x', 'y', 'z', 'u', 'v', 'w', 'fx', 'fy', 'fz',
             'rho', 'm', 'h', 'p', 'tag', 'pid', 'gid']
        )

        boundary.add_property('V')
        boundary.add_property('fx')
        boundary.add_property('fy')
        boundary.add_property('fz')

        return [fluid, boundary, obstacle]

    def create_solver(self):
        kernel = WendlandQuintic(dim=dim)

        integrator = EPECIntegrator(fluid=WCSPHStep(),
                                    obstacle=RK2StepRigidBody(),
                                    boundary=WCSPHStep())
        solver = Solver(kernel=kernel, dim=dim, integrator=integrator,
                        tf=tf, dt=dt, adaptive_timestep=True, n_damp=0)
        return solver

    def create_equations(self):
        co = 10.0 * self.geom.get_max_speed(g=9.81)

        gamma = 7.0
        alpha = 0.5
        beta = 0.0

        equations = [

            Group(equations=[
                BodyForce(dest='obstacle', sources=None, gz=-9.81),
                NumberDensity(dest='obstacle', sources=['obstacle']),
                NumberDensity(dest='boundary', sources=['boundary']),
            ], ),

            # Equation of state
            Group(equations=[

                TaitEOS(
                    dest='fluid', sources=None, rho0=rho0, c0=co,
                    gamma=gamma
                ),
                TaitEOSHGCorrection(
                    dest='boundary', sources=None, rho0=rho0, c0=co,
                    gamma=gamma
                ),
                TaitEOSHGCorrection(
                    dest='obstacle', sources=None, rho0=rho0, c0=co,
                    gamma=gamma
                ),

            ], real=False),

            # Continuity, momentum and xsph equations
            Group(equations=[
                ContinuityEquation(
                    dest='fluid', sources=['fluid', 'boundary', 'obstacle']
                ),
                ContinuityEquation(dest='boundary', sources=['fluid']),
                ContinuityEquation(dest='obstacle', sources=['fluid']),

                MomentumEquation(dest='fluid',
                                 sources=['fluid', 'boundary'],
                                 alpha=alpha, beta=beta, gz=-9.81, c0=co,
                                 tensile_correction=True),

                PressureRigidBody(
                    dest='fluid', sources=['obstacle'], rho0=rho0
                ),

                XSPHCorrection(dest='fluid', sources=['fluid']),

                RigidBodyForceGPUGems(
                    dest='obstacle', sources=['boundary'], k=1.0, d=2.0,
                    eta=0.1, kt=0.1
                ),

            ]),
            Group(equations=[RigidBodyMoments(dest='obstacle', sources=None)]),
            Group(equations=[RigidBodyMotion(dest='obstacle', sources=None)]),

        ]
        return equations

if __name__ == '__main__':
    app = DamBreak3DSPH()
    app.run()

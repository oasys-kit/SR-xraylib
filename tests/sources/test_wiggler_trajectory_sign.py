#
# Sign convention of srfunc.wiggler_trajectory: electrons (charge -|e|, Lorentz force F = -|e| v x B), in the SHADOW
# frame (y along the beam, z up, x toward the interior of a ring turning clockwise seen from above):
#     d betax / dy = d^2 x / dy^2 = -|e| B_z / (gamma m_e c)
# i.e., B_z < 0 deflects toward +x (as for the SHADOW4 bending magnet), and the curvature (column 6 of the
# trajectory) is d^2 x / dy^2.
#
import unittest
import numpy
import scipy.constants as codata
from srxraylib.sources.srfunc import wiggler_trajectory

E_GEV = 6.04
GAMMA = 1e9 * E_GEV / (codata.m_e * codata.c ** 2 / codata.e)


def second_derivative(x, y):
    return numpy.gradient(numpy.gradient(x, y), y)


class TestWigglerTrajectorySign(unittest.TestCase):

    def test_single_pole_from_field(self):
        y = numpy.linspace(-0.5, 0.5, 2001)
        for B0 in [-0.8, 0.8]:
            B = B0 * numpy.exp(-(y / 0.1) ** 2)
            traj, pars = wiggler_trajectory(b_from=1, inData=numpy.vstack((y, B)).T, nPer=1, ener_gev=E_GEV)
            x, yy, betax, curv, b = traj[0], traj[1], traj[3], traj[6], traj[7]
            # deflection: toward +x for B < 0
            self.assertEqual(numpy.sign(x[-1] - x[0]), -numpy.sign(B0))
            self.assertEqual(numpy.sign(betax[-1]), -numpy.sign(B0))
            # curvature = d2x/dy2 = -|e| B / (gamma m c)
            pole = numpy.abs(b) > 0.5 * numpy.abs(b).max()
            numpy.testing.assert_allclose(curv[pole] / b[pole], -codata.e / (GAMMA * codata.m_e * codata.c), rtol=1e-3)
            numpy.testing.assert_allclose(second_derivative(x, yy)[pole], curv[pole], rtol=2e-2)

    def test_sinusoidal(self):
        traj, pars = wiggler_trajectory(b_from=0, kValue=20.0, per=0.15, nPer=6, nTrajPoints=501, ener_gev=E_GEV)
        x, yy, curv, b = traj[0], traj[1], traj[6], traj[7]
        pole = numpy.abs(b) > 0.5 * numpy.abs(b).max()
        self.assertTrue(numpy.all(numpy.sign(curv[pole]) == -numpy.sign(b[pole])))
        self.assertTrue(numpy.all(numpy.sign(second_derivative(x, yy)[pole]) == numpy.sign(curv[pole])))


if __name__ == "__main__":
    unittest.main()

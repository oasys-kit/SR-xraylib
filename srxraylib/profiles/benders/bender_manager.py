#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2023. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #

import numpy
import decimal
from scipy.optimize import root

from srxraylib.profiles.benders.bender_io import BenderOuputData, BenderFitParameters, BenderStructuralParameters, BenderMovement

def get_significant_digits(number):
    return abs(decimal.Decimal(str(number)).as_tuple().exponent)


class AbstractBenderManager:
    def __init__(self, bender_structural_parameters : BenderStructuralParameters):
        self._bender_structural_parameters = bender_structural_parameters

    @property
    def bender_structural_parameters(self): return self._bender_structural_parameters

    def fit_bender_at_focus_position(self, bender_fit_parameters: BenderFitParameters) -> BenderOuputData: raise NotImplementedError
    def get_bender_shape_from_movement(self, bender_movement: BenderMovement) ->  BenderOuputData: raise NotImplementedError

    def get_q_upstream(self, bender_movement : BenderMovement):   raise NotImplementedError
    def get_q_downstream(self, bender_movement : BenderMovement): raise NotImplementedError
    def get_q_ideal_surface(self, bender_movement : BenderMovement):
        return 0.5*(self.get_q_upstream(bender_movement)+self.get_q_downstream(bender_movement))

    def calculate_ideal_surface(self, q=None, sign=-1):
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = self.get_conic_coefficients(q)

        x = numpy.linspace(-self.bender_structural_parameters.dim_x_minus, self.bender_structural_parameters.dim_x_plus, self.bender_structural_parameters.bender_bin_x + 1)
        y = numpy.linspace(-self.bender_structural_parameters.dim_y_minus, self.bender_structural_parameters.dim_y_plus, self.bender_structural_parameters.bender_bin_y + 1)

        X, Y = numpy.meshgrid(x, y)

        def equation_to_solve(Z):
            return c1 * (X ** 2) + c2 * (Y ** 2) + c3 * (Z ** 2) + c4 * X * Y + c5 * Y * Z + c6 * X * Z + c7 * X + c8 * Y + c9 * Z + c10

        z_start = numpy.zeros(X.shape)
        result = root(equation_to_solve, z_start, method='df-sane', tol=None)

        z = result.x if result.success else z_start

        return x, y, z.T

    def get_conic_coefficients(self, q=None):
        theta_grazing = self.bender_structural_parameters.grazing_angle
        ssour         = self.bender_structural_parameters.p
        simag         = q if not q is None else self.bender_structural_parameters.q

        theta = (numpy.pi / 2) - theta_grazing

        ax_maj = (ssour + simag) / 2
        ax_min = numpy.sqrt(simag * ssour) * numpy.cos(theta)
        eccentricity = numpy.sqrt(ax_maj ** 2 - ax_min ** 2) / ax_maj
        #
        # The center is computed on the basis of the object and image positions
        #
        y_center = (ssour - simag) * 0.5 / eccentricity
        z_center = -numpy.sqrt(1 - y_center ** 2 / ax_maj ** 2) * ax_min
        #
        # Computes now the normal in the mirror center.
        #
        rn_center = numpy.zeros(3)
        rn_center[0] = 0.0
        rn_center[1] = -2 * y_center / ax_maj ** 2
        rn_center[2] = -2 * z_center / ax_min ** 2
        rn_center /= numpy.sqrt((rn_center ** 2).sum())
        #
        # Computes the tangent versor in the mirror center.
        #
        rt_center = numpy.zeros(3)
        rt_center[0] = 0.0
        rt_center[1] = rn_center[2]
        rt_center[2] = -rn_center[1]

        # Computes now the quadric coefficient with the mirror center
        # located at (0,0,0) and normal along (0,0,1)

        A = 1 / ax_min ** 2
        B = 1 / ax_maj ** 2
        C = A

        c1 = 0.0  # A if ellipsoid and 0 if cylinder -> managed benders are only cylinders
        c2 = B * rt_center[1] ** 2 + C * rt_center[2] ** 2
        c3 = B * rn_center[1] ** 2 + C * rn_center[2] ** 2
        c4 = 0.0
        c5 = 2 * (B * rn_center[1] * rt_center[1] + C * rn_center[2] * rt_center[2])
        c6 = 0.0
        c7 = 0.0
        c8 = 0.0
        c9 = 2 * (B * y_center * rn_center[1] + C * z_center * rn_center[2])
        c10 = 0.0

        return c1, c2, c3, c4, c5, c6, c7, c8, c9, c10

class CalibrationParameters:
    __parameters_upstream   = [0.0, 0.0]
    __parameters_downstream = [0.0, 0.0]

    def __init__(self, parameters_upstream, parameters_downstream):
        self.__parameters_upstream   = parameters_upstream
        self.__parameters_downstream = parameters_downstream

    @property
    def upstream(self):   return self.__parameters_upstream
    @property
    def downstream(self): return self.__parameters_downstream

class StandardBenderManager(AbstractBenderManager):
    def __init__(self, bender_structural_parameters : BenderStructuralParameters):
        super(StandardBenderManager, self).__init__(bender_structural_parameters=bender_structural_parameters)

    def get_q_upstream(self, bender_movement : BenderMovement):   return bender_movement.position_upstream # position coincides with q -> direct calculation
    def get_q_downstream(self, bender_movement : BenderMovement): return bender_movement.position_downstream

class CalibratedBenderManager(AbstractBenderManager):
    def __init__(self, bender_structural_parameters : BenderStructuralParameters, calibration_parameters : CalibrationParameters):
        super(CalibratedBenderManager, self).__init__(bender_structural_parameters=bender_structural_parameters)
        self._calibration_parameters       = calibration_parameters


    def get_q_upstream(self, bender_movement : BenderMovement):
        if bender_movement.position_upstream is None: return None
        return CalibratedBenderManager.__get_q_from_calibration(bender_movement.position_upstream, self._calibration_parameters.upstream)

    def get_q_downstream(self, bender_movement : BenderMovement):
        if bender_movement.position_downstream is None: return None
        return CalibratedBenderManager.__get_q_from_calibration(bender_movement.position_downstream, self._calibration_parameters.downstream)

    @classmethod # 1/q = p0*pos + p1
    def __get_q_from_calibration(cls, position, parameters):  return 1 / (parameters[0] * position + parameters[1])
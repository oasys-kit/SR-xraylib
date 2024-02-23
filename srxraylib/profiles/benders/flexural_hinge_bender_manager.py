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
import copy

import numpy
from scipy.interpolate import interp2d, make_interp_spline
from scipy.optimize import curve_fit

from srxraylib.profiles.benders.bender_io import BenderMovement, BenderFitParameters, BenderStructuralParameters, BenderOuputData
from srxraylib.profiles.benders.bender_manager import StandardBenderManager, CalibratedBenderManager, CalibrationParameters, get_significant_digits

class FlexuralHingeBenderFitParameters(BenderFitParameters):
    def __init__(self,
                 optimized_length=None,
                 n_fit_steps=None,
                 M1=None,
                 M1_min=None,
                 M1_max=None,
                 M1_fixed=False,
                 e=None,
                 e_min=None,
                 e_max=None,
                 e_fixed=False,
                 ratio=None,
                 ratio_min=None,
                 ratio_max=None,
                 ratio_fixed=False):
        super(FlexuralHingeBenderFitParameters, self).__init__(optimized_length=optimized_length,
                                                               n_fit_steps=n_fit_steps)
        self.M1 = M1
        self.M1_max = M1_max
        self.M1_min = M1_min
        self.M1_fixed = M1_fixed
        self.e = e
        self.e_max = e_max
        self.e_min = e_min
        self.e_fixed = e_fixed
        self.ratio = ratio
        self.ratio_max = ratio_max
        self.ratio_min = ratio_min
        self.ratio_fixed = ratio_fixed

class FlexuralHingeBenderOuputData(BenderOuputData):
    def __init__(self,
                 x=None,
                 y=None,
                 ideal_profile=None,
                 bender_profile=None,
                 correction_profile=None,
                 titles=None,
                 z_bender_correction=None,
                 z_figure_error=None,
                 z_bender_correction_no_figure_error=None,
                 M1_out=None,
                 e_out=None,
                 ratio_out=None):
        super(FlexuralHingeBenderOuputData, self).__init__(x,
                                                 y,
                                                 ideal_profile,
                                                 bender_profile,
                                                 correction_profile,
                                                 titles,
                                                 z_bender_correction,
                                                 z_figure_error,
                                                 z_bender_correction_no_figure_error)
        self.M1_out    = M1_out
        self.e_out     = e_out
        self.ratio_out = ratio_out

class FlexuralHingeBenderStructuralParameters(BenderStructuralParameters):
    def __init__(self,
                 dim_x_minus=None,
                 dim_x_plus=None,
                 bender_bin_x=None,
                 dim_y_minus=None,
                 dim_y_plus=None,
                 bender_bin_y=None,
                 p=None,
                 q=None,
                 grazing_angle=None,
                 E=None,
                 h=None,
                 figure_error_mesh=None,
                 shape=None,
                 bender_type=None,
                 M1=None,
                 e=None,
                 ratio=None,
                 workspace_units_to_m=None,
                 workspace_units_to_mm=None):
        super(FlexuralHingeBenderStructuralParameters, self).__init__(dim_x_minus=dim_x_minus,
                                                                      dim_x_plus=dim_x_plus ,
                                                                      bender_bin_x=bender_bin_x ,
                                                                      dim_y_minus=dim_y_minus,
                                                                      dim_y_plus=dim_y_plus,
                                                                      bender_bin_y=bender_bin_y,
                                                                      p=p,
                                                                      q=q,
                                                                      grazing_angle=grazing_angle,
                                                                      E=E,
                                                                      h=h,
                                                                      figure_error_mesh=figure_error_mesh,
                                                                      workspace_units_to_m=workspace_units_to_m,
                                                                      workspace_units_to_mm=workspace_units_to_mm)
        self.shape       = shape
        self.bender_type = bender_type
        self.M1          = M1
        self.e           = e
        self.ratio       = ratio

class MirrorShape:
    TRAPEZIUM = 0
    RECTANGLE = 1

class BenderType:
    SINGLE_MOMENTUM = 0
    DOUBLE_MOMENTUM = 1

epsilon_minus = 1 - 1e-8
epsilon_plus = 1 + 1e-8

# Decorator
class _FlexuralHingeBenderCalculator():
    def __init__(self, bender_manager):
        self.__bender_manager = bender_manager

    def fit_bender_at_focus_position(self, bender_fit_parameters: FlexuralHingeBenderFitParameters) -> FlexuralHingeBenderOuputData:
        workspace_units_to_m  = self.__bender_manager.bender_structural_parameters.workspace_units_to_m
        workspace_units_to_mm = self.__bender_manager.bender_structural_parameters.workspace_units_to_mm
        shape                 = self.__bender_manager.bender_structural_parameters.shape
        bender_type           = self.__bender_manager.bender_structural_parameters.bender_type
        optimized_length      = bender_fit_parameters.optimized_length

        ideal_surface_coords = self.__bender_manager.calculate_ideal_surface()

        bender_profile, parameters, ideal_profile, cursor = self.__fit_bender_parameters(bender_fit_parameters, ideal_surface_coords)

        # rotate back to Shadow system
        bender_data = self.__generate_bender_output_data(ideal_profile, bender_profile, ideal_surface_coords)

        correction_profile = bender_data.correction_profile

        # r-squared = 1 - residual sum of squares / total sum of squares
        r_squared = 1 - (numpy.sum(correction_profile ** 2) / numpy.sum((ideal_profile - numpy.mean(ideal_profile)) ** 2))
        rms       = round(correction_profile.std() * 1e9 * workspace_units_to_m, 6)
        if not bender_fit_parameters.optimized_length is None:  rms_opt = round(correction_profile[cursor].std() * 1e9 * workspace_units_to_m, 6)

        bender_data.titles = ["Bender vs. Ideal Profiles" + "\n" + r'$R^2$ = ' + str(r_squared),
                              "Correction Profile 1D, r.m.s. = " + str(rms) + " nm" +
                              ("" if optimized_length is None else (", " + str(rms_opt) + " nm (optimized)"))]

        bender_data.M1_out = round(parameters[0], int(3 + get_significant_digits(workspace_units_to_mm)))
        if shape == MirrorShape.TRAPEZIUM:
            bender_data.e_out = round(parameters[1], 5)
            if bender_type == BenderType.DOUBLE_MOMENTUM: bender_data.ratio_out = round(parameters[2], 5)
        elif shape == MirrorShape.RECTANGLE:
            if bender_type == BenderType.DOUBLE_MOMENTUM: bender_data.ratio_out = round(parameters[1], 5)

        # set the structure of the mirror at focus
        self.__bender_manager.bender_structural_parameters.M1 = bender_data.M1_out
        if shape == MirrorShape.TRAPEZIUM:            self.__bender_manager.bender_structural_parameters.e     = bender_data.e_out
        if bender_type == BenderType.DOUBLE_MOMENTUM: self.__bender_manager.bender_structural_parameters.ratio = bender_data.ratio_out

        return bender_data

    def get_bender_shape_from_movement(self, bender_movement: BenderMovement) ->  FlexuralHingeBenderOuputData:
        shape                 = self.__bender_manager.bender_structural_parameters.shape
        bender_type           = self.__bender_manager.bender_structural_parameters.bender_type
        workspace_units_to_mm = self.__bender_manager.bender_structural_parameters.workspace_units_to_mm
        workspace_units_to_m  = self.__bender_manager.bender_structural_parameters.workspace_units_to_m
        L                     = self.__bender_manager.bender_structural_parameters.dim_y_plus + self.__bender_manager.bender_structural_parameters.dim_y_minus

        # ORIGINAL ELLIPSE: IDEAL FOCUSING ON THE SAMPLE
        ideal_surface_coords  = self.__bender_manager.calculate_ideal_surface(q=self.__bender_manager.bender_structural_parameters.q)

        _, y, _ = ideal_surface_coords
        ideal_profile = self.__get_ideal_profile(L, ideal_surface_coords)

        q_downstream = self.__bender_manager.get_q_downstream(bender_movement) # M1
        q_upstream   = self.__bender_manager.get_q_upstream(bender_movement)   # M2 -> None for single momentum benders

        bender_fit_parameters = self.__get_fit_parameters_for_movement()

        if bender_type == BenderType.SINGLE_MOMENTUM:
            ideal_surface_coords_downstream = self.__bender_manager.calculate_ideal_surface(q=q_downstream)

            if not q_upstream is None: raise ValueError("Specify q-downstream only on a Single Momentum Bender")

            bender_profile, parameters, ideal_profile, _ = self.__fit_bender_parameters(bender_fit_parameters, ideal_surface_coords_downstream)

            M1 = parameters[0]
        else:
            ideal_surface_coords_downstream = self.__bender_manager.calculate_ideal_surface(q=q_downstream)
            ideal_surface_coords_upstream   = self.__bender_manager.calculate_ideal_surface(q=q_upstream)

            _, parameters_downstream, _, _ = self.__fit_bender_parameters(bender_fit_parameters, ideal_surface_coords_downstream)

            M1 = parameters_downstream[0]
            bender_fit_parameters.M1 = M1

            _, parameters_upstream, _, _   = self.__fit_bender_parameters(bender_fit_parameters, ideal_surface_coords_upstream)

            if shape == MirrorShape.TRAPEZIUM:   ratio = parameters_upstream[2]
            elif shape == MirrorShape.RECTANGLE: ratio = parameters_upstream[1]

            M2 = parameters_upstream[0] * ratio

            E  = self.__bender_manager.bender_structural_parameters.E
            h  = self.__bender_manager.bender_structural_parameters.h
            b0 = self.__bender_manager.bender_structural_parameters.dim_x_plus + self.__bender_manager.bender_structural_parameters.dim_x_minus

            bender_profile = self.__general_bender_function(Y=y*workspace_units_to_mm,
                                                            M1=M1*workspace_units_to_mm,
                                                            e=bender_fit_parameters.e,
                                                            ratio=M2/M1,
                                                            E=E/workspace_units_to_mm**2,
                                                            h=h*workspace_units_to_mm,
                                                            b0=b0*workspace_units_to_mm,
                                                            L=L*workspace_units_to_mm)
            bender_profile /= workspace_units_to_mm

        bender_data = self.__generate_bender_output_data(ideal_profile, bender_profile, ideal_surface_coords)

        bender_data.M1_out = round(M1, int(3 + get_significant_digits(workspace_units_to_mm)))
        if bender_type == BenderType.DOUBLE_MOMENTUM: bender_data.ratio_out = M2/M1

        correction_profile = bender_data.correction_profile

        r_squared = 1 - (numpy.sum(correction_profile ** 2) / numpy.sum((ideal_profile - numpy.mean(ideal_profile)) ** 2))
        rms       = round(correction_profile.std() * 1e9 * workspace_units_to_m, 6)

        bender_data.titles = ["Bender vs. Ideal Profiles" + "\n" + r'$R^2$ = ' + str(r_squared),
                              "Correction Profile 1D, r.m.s. = " + str(rms) + " nm"]

        return bender_data

    def __generate_bender_output_data(self, ideal_profile, bender_profile, ideal_surface_coords):
        x, y, z = ideal_surface_coords

        # back to Shadow Axis system: upstream is positive
        bender_profile = bender_profile[::-1]
        ideal_profile  = ideal_profile[::-1]

        correction_profile = ideal_profile - bender_profile

        z_bender_correction_no_figure_error = numpy.tile(correction_profile, (z.shape[0], 1))
        z_bender_correction_no_figure_error.reshape(z.shape)

        if not self.__bender_manager.bender_structural_parameters.figure_error_mesh is None:
            x_e, y_e, z_e = self.__bender_manager.bender_structural_parameters.figure_error_mesh

            if len(x) == len(x_e) and len(y) == len(y_e) and \
                    x[0] == x_e[0] and x[-1] == x_e[-1] and \
                    y[0] == y_e[0] and y[-1] == y_e[-1]:
                z_figure_error = z_e
            else:
                z_figure_error = interp2d(y_e, x_e, z_e, kind='cubic')(y, x)

            z_bender_correction = z_bender_correction_no_figure_error + z_figure_error
        else:
            z_figure_error = None
            z_bender_correction = z_bender_correction_no_figure_error

        return FlexuralHingeBenderOuputData(x=x,
                                            y=y,
                                            ideal_profile=ideal_profile,  # 1D
                                            bender_profile=bender_profile,
                                            correction_profile=correction_profile,
                                            z_bender_correction=z_bender_correction,
                                            z_figure_error=z_figure_error,
                                            z_bender_correction_no_figure_error=z_bender_correction_no_figure_error)

    def __get_fit_parameters_for_movement(self):
        M1 = self.__bender_manager.bender_structural_parameters.M1

        return FlexuralHingeBenderFitParameters(optimized_length=None,
                                                n_fit_steps=5,
                                                M1=M1,
                                                M1_min=0,
                                                M1_max=M1*10,
                                                M1_fixed=False,
                                                e=self.__bender_manager.bender_structural_parameters.e,
                                                e_min=0.0,
                                                e_max=0.0,
                                                e_fixed=True,
                                                ratio=self.__bender_manager.bender_structural_parameters.ratio,
                                                ratio_min=0.0,
                                                ratio_max=10.0,
                                                ratio_fixed=False)

    def __fit_bender_parameters(self, bender_fit_parameters, ideal_surface_coords):
        _, y, z = ideal_surface_coords

        E                     = self.__bender_manager.bender_structural_parameters.E
        h                     = self.__bender_manager.bender_structural_parameters.h
        shape                 = self.__bender_manager.bender_structural_parameters.shape
        bender_type           = self.__bender_manager.bender_structural_parameters.bender_type
        optimized_length      = bender_fit_parameters.optimized_length
        n_fit_steps           = bender_fit_parameters.n_fit_steps
        workspace_units_to_mm = self.__bender_manager.bender_structural_parameters.workspace_units_to_mm

        b0 = self.__bender_manager.bender_structural_parameters.dim_x_plus + self.__bender_manager.bender_structural_parameters.dim_x_minus
        L  = self.__bender_manager.bender_structural_parameters.dim_y_plus + self.__bender_manager.bender_structural_parameters.dim_y_minus

        # flip the coordinate system to be consistent with Mike's formulas
        ideal_profile = z[0, :][::-1]  # one row is the profile of the cylinder, enough for the minimizer
        ideal_profile += -ideal_profile[0] + ((L / 2 + y) * (ideal_profile[0] - ideal_profile[-1])) / L  # Rotation

        # in units different from mm, there is problem of precision and the fitter struggles
        E                            /= workspace_units_to_mm ** 2
        h                            *= workspace_units_to_mm
        b0                           *= workspace_units_to_mm
        L                            *= workspace_units_to_mm
        bender_fit_parameters.M1     *= workspace_units_to_mm
        bender_fit_parameters.M1_min *= workspace_units_to_mm
        bender_fit_parameters.M1_max *= workspace_units_to_mm

        if optimized_length is None:
            cursor            = None
            y_fit             = copy.deepcopy(y)
            ideal_profile_fit = copy.deepcopy(ideal_profile)
        else:
            cursor            = numpy.where(numpy.logical_and(y >= -optimized_length / 2, y <= optimized_length / 2))
            y_fit             = copy.deepcopy(y[cursor])
            ideal_profile_fit = copy.deepcopy(ideal_profile[cursor])

        y_fit             *= workspace_units_to_mm
        ideal_profile_fit *= workspace_units_to_mm

        initial_guess   = None
        constraints     = None
        bender_function = None

        if shape == MirrorShape.TRAPEZIUM:
            def bender_function_2m(Y, M1, e, ratio):
                return self.__general_bender_function(Y, M1, e, ratio, E, h, b0, L)

            def bender_function_1m(Y, M1, e):
                return self.__general_bender_function(Y, M1, e, 1.0, E, h, b0, L)

            if bender_type == BenderType.SINGLE_MOMENTUM:
                bender_function = bender_function_1m
                initial_guess = [bender_fit_parameters.M1, bender_fit_parameters.e]
                constraints = [[bender_fit_parameters.M1_min if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_minus),
                                bender_fit_parameters.e_min if bender_fit_parameters.e_fixed == False else (bender_fit_parameters.e * epsilon_minus)],
                               [bender_fit_parameters.M1_max if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_plus),
                                bender_fit_parameters.e_max if bender_fit_parameters.e_fixed == False else (bender_fit_parameters.e * epsilon_plus)]]
            elif bender_type == BenderType.DOUBLE_MOMENTUM:
                bender_function = bender_function_2m
                initial_guess = [bender_fit_parameters.M1, bender_fit_parameters.e, bender_fit_parameters.ratio]
                constraints = [[bender_fit_parameters.M1_min if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_minus),
                                bender_fit_parameters.e_min if bender_fit_parameters.e_fixed == False else (bender_fit_parameters.e * epsilon_minus),
                                bender_fit_parameters.ratio_min if bender_fit_parameters.ratio_fixed == False else (bender_fit_parameters.ratio * epsilon_minus)],
                               [bender_fit_parameters.M1_max if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_plus),
                                bender_fit_parameters.e_max if bender_fit_parameters.e_fixed == False else (bender_fit_parameters.e * epsilon_plus),
                                bender_fit_parameters.ratio_max if bender_fit_parameters.ratio_fixed == False else (bender_fit_parameters.ratio * epsilon_plus)]]
        elif shape == MirrorShape.RECTANGLE:
            def bender_function_2m(Y, M1, ratio):
                return self.__general_bender_function(Y, M1, 0.0, ratio, E, h, b0, L)

            def bender_function_1m(Y, M1):
                return self.__general_bender_function(Y, M1, 0.0, 1.0, E, h, b0, L)

            if bender_type == BenderType.SINGLE_MOMENTUM:
                bender_function = bender_function_1m
                initial_guess = [bender_fit_parameters.M1]
                constraints = [[bender_fit_parameters.M1_min if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_minus)],
                               [bender_fit_parameters.M1_max if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_plus)]]
            elif bender_type == BenderType.DOUBLE_MOMENTUM:
                bender_function = bender_function_2m
                initial_guess = [bender_fit_parameters.M1, bender_fit_parameters.ratio]
                constraints = [[bender_fit_parameters.M1_min if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_minus),
                                bender_fit_parameters.ratio_min if bender_fit_parameters.ratio_fixed == False else (bender_fit_parameters.ratio * epsilon_minus)],
                               [bender_fit_parameters.M1_max if bender_fit_parameters.M1_fixed == False else (bender_fit_parameters.M1 * epsilon_plus),
                                bender_fit_parameters.ratio_max if bender_fit_parameters.ratio_fixed == False else (bender_fit_parameters.ratio * epsilon_plus)]]

        for i in range(n_fit_steps):
            parameters, _ = curve_fit(f=bender_function,
                                      xdata=y_fit,
                                      ydata=ideal_profile_fit,
                                      p0=initial_guess,
                                      bounds=constraints,
                                      method='trf')
            initial_guess = parameters

        if len(parameters) == 1:   bender_profile = bender_function(y*workspace_units_to_mm, parameters[0])
        elif len(parameters) == 2: bender_profile = bender_function(y*workspace_units_to_mm, parameters[0], parameters[1])
        else:                      bender_profile = bender_function(y*workspace_units_to_mm, parameters[0], parameters[1], parameters[2])

        # restore to user units
        parameters[0]                /= workspace_units_to_mm
        bender_profile               /= workspace_units_to_mm
        bender_fit_parameters.M1     /= workspace_units_to_mm
        bender_fit_parameters.M1_min /= workspace_units_to_mm
        bender_fit_parameters.M1_max /= workspace_units_to_mm

        return bender_profile, parameters, ideal_profile, cursor

    @classmethod
    def __get_ideal_profile(cls, L, ideal_surface_coords) -> numpy.ndarray:
        x, y, z = ideal_surface_coords

        ideal_profile = z[0, :][::-1]  # one row is the profile of the cylinder, enough for the minimizer
        ideal_profile += -ideal_profile[0] + ((L / 2 + y) * (ideal_profile[0] - ideal_profile[-1])) / L  # Rotation

        return ideal_profile

    @classmethod    
    def __general_bender_function(cls, Y, M1, e, ratio, E, h, b0, L):
        '''

        :param Y: tangential coordinates
        :param M1: DOWNSTREAM MOMENTUM -> applied to the short side of the trapezium
        :param e: long side = (1 + e)b0
        :param ratio: M2/M1 -> M2: UPSTREAM MOMENTUM
        :param E: young's modulus
        :param h: thickness
        :param b0: short side
        :param L: total length of the mirror
        :return: bender profile(Y)
        '''

        Eh_3 = E * h ** 3
    
        M2 = M1 * ratio
        A = (M1 + M2) / 2
        B = (M1 - M2) / L
        C = Eh_3 * (2 * b0 + e * b0) / 24
        D = Eh_3 * e * b0 / (12 * L)
        H = (A * D + B * C) / D ** 2
        CDLP = C + D * L / 2
        CDLM = C - D * L / 2
        F = (H / L) * ((CDLM * numpy.log(CDLM) - CDLP * numpy.log(CDLP)) / D + L)
        G = (-H * ((CDLM * numpy.log(CDLM) + CDLP * numpy.log(CDLP))) + (B * L ** 2) / 4) / (2 * D)
        CDY = C + D * Y
    
        return H * ((CDY / D) * numpy.log(CDY) - Y) - (B * Y ** 2) / (2 * D) + F * Y + G


class FlexuralHingeStandardBenderManager(StandardBenderManager):
    def __init__(self,
                 bender_structural_parameters : FlexuralHingeBenderStructuralParameters):
        super(FlexuralHingeStandardBenderManager, self).__init__(bender_structural_parameters=bender_structural_parameters)
        self.__calculator = _FlexuralHingeBenderCalculator(bender_manager=self)

    def fit_bender_at_focus_position(self, bender_fit_parameters: FlexuralHingeBenderFitParameters) -> FlexuralHingeBenderOuputData:
        return self.__calculator.fit_bender_at_focus_position(bender_fit_parameters)

    def get_bender_shape_from_movement(self, bender_movement: BenderMovement) ->  FlexuralHingeBenderOuputData:
        return self.__calculator.get_bender_shape_from_movement(bender_movement)


class FlexuralHingeCalibratedBenderManager(CalibratedBenderManager):
    def __init__(self,
                 bender_structural_parameters: FlexuralHingeBenderStructuralParameters,
                 calibration_parameters : CalibrationParameters):
        super(FlexuralHingeCalibratedBenderManager, self).__init__(bender_structural_parameters=bender_structural_parameters,
                                                                   calibration_parameters=calibration_parameters)
        self.__calculator = _FlexuralHingeBenderCalculator(bender_manager=self)

    def fit_bender_at_focus_position(self, bender_fit_parameters: FlexuralHingeBenderFitParameters) -> FlexuralHingeBenderOuputData:
        return self.__calculator.fit_bender_at_focus_position(bender_fit_parameters)

    def get_bender_shape_from_movement(self, bender_movement: BenderMovement) -> FlexuralHingeBenderOuputData:
        return self.__calculator.get_bender_shape_from_movement(bender_movement)






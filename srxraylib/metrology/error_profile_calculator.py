#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2021, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2021. UChicago Argonne, LLC. This software was produced       #
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

from srxraylib.metrology import profiles_simulation

class ErrorProfileInputParameters():
    def __init__(self, widget=None):
        if not widget is None:
            self.si_to_user_units              = widget.si_to_user_units
            self.kind_of_profile_x             = widget.kind_of_profile_x
            self.kind_of_profile_y             = widget.kind_of_profile_y
            self.step_x                        = widget.step_x
            self.step_y                        = widget.step_y
            self.dimension_x                   = widget.dimension_x
            self.dimension_y                   = widget.dimension_y
            self.power_law_exponent_beta_x     = widget.power_law_exponent_beta_x
            self.power_law_exponent_beta_y     = widget.power_law_exponent_beta_y
            self.correlation_length_x          = widget.correlation_length_x
            self.correlation_length_y          = widget.correlation_length_y
            self.error_type_x                  = widget.error_type_x
            self.error_type_y                  = widget.error_type_y
            self.rms_x                         = widget.rms_x
            self.montecarlo_seed_x             = widget.montecarlo_seed_x
            self.rms_y                         = widget.rms_y
            self.montecarlo_seed_y             = widget.montecarlo_seed_y
            self.heigth_profile_1D_file_name_x = widget.heigth_profile_1D_file_name_x
            self.delimiter_x                   = widget.delimiter_x
            self.conversion_factor_x_x         = widget.conversion_factor_x_x
            self.conversion_factor_x_y         = widget.conversion_factor_x_y
            self.center_x                      = widget.center_x
            self.modify_x                      = widget.modify_x
            self.new_length_x                  = widget.new_length_x
            self.filler_value_x                = widget.filler_value_x
            self.renormalize_x                 = widget.renormalize_x
            self.heigth_profile_1D_file_name_y = widget.heigth_profile_1D_file_name_y
            self.delimiter_y                   = widget.delimiter_y
            self.conversion_factor_y_x         = widget.conversion_factor_y_x
            self.conversion_factor_y_y         = widget.conversion_factor_y_y
            self.center_y                      = widget.center_y
            self.modify_y                      = widget.modify_y
            self.new_length_y                  = widget.new_length_y
            self.filler_value_y                = widget.filler_value_y
            self.renormalize_y                 = widget.renormalize_y
        else:
            self.si_to_user_units = 1e-3
            self.kind_of_profile_x = 0
            self.kind_of_profile_y = 0
            self.step_x = 0.0
            self.step_y = 0.0
            self.dimension_x = 0.0
            self.dimension_y = 0.0
            self.power_law_exponent_beta_x = 3.0
            self.power_law_exponent_beta_y = 3.0
            self.correlation_length_x = 0.3
            self.correlation_length_y = 0.3
            self.error_type_x = profiles_simulation.FIGURE_ERROR
            self.error_type_y = profiles_simulation.FIGURE_ERROR
            self.rms_x = 0.1
            self.montecarlo_seed_x = 8787
            self.rms_y = 1
            self.montecarlo_seed_y = 8788
            self.heigth_profile_1D_file_name_x = "mirror_1D_x.dat"
            self.delimiter_x = 0
            self.conversion_factor_x_x = 0.001
            self.conversion_factor_x_y = 1e-6
            self.center_x = 1
            self.modify_x = 0
            self.new_length_x = 0.0
            self.filler_value_x = 0.0
            self.renormalize_x = 0
            self.heigth_profile_1D_file_name_y = "mirror_1D_y.dat"
            self.delimiter_y = 0
            self.conversion_factor_y_x = 0.001
            self.conversion_factor_y_y = 1e-6
            self.center_y = 1
            self.modify_y = 0
            self.new_length_y = 0.0
            self.filler_value_y = 0.0
            self.renormalize_y = 0
            
class DabamInputParameters():


    def __init__(self, dabam_server, widget=None):
        if not widget is None:
            self.server                        = dabam_server
            self.si_to_user_units              = widget.si_to_user_units
            self.use_undetrended               = widget.use_undetrended
            self.step_x                        = widget.step_x
            self.dimension_x                   = widget.dimension_x
            self.kind_of_profile_x             = widget.kind_of_profile_x
            self.power_law_exponent_beta_x     = widget.power_law_exponent_beta_x
            self.correlation_length_x          = widget.correlation_length_x
            self.error_type_x                  = widget.error_type_x
            self.rms_x                         = widget.rms_x
            self.montecarlo_seed_x             = widget.montecarlo_seed_x
            self.heigth_profile_1D_file_name_x = widget.heigth_profile_1D_file_name_x
            self.delimiter_x                   = widget.delimiter_x
            self.conversion_factor_x_x         = widget.conversion_factor_x_x
            self.conversion_factor_x_y         = widget.conversion_factor_x_y
            self.center_x                      = widget.center_x
            self.modify_x                      = widget.modify_x
            self.new_length_x                  = widget.new_length_x
            self.filler_value_x                = widget.filler_value_x
            self.renormalize_x                 = widget.renormalize_x
            self.center_y                      = widget.center_y
            self.modify_y                      = widget.modify_y
            self.new_length_y                  = widget.new_length_y
            self.filler_value_y                = widget.filler_value_y
            self.renormalize_y                 = widget.renormalize_y
            self.error_type_y                  = widget.error_type_y
            self.rms_y                         = widget.rms_y
        else:
            self.server = dabam_server
            self.si_to_user_units = 1e-3
            self.use_undetrended = 0
            self.step_x = 0.0
            self.dimension_x = 0.0
            self.kind_of_profile_x = 3
            self.power_law_exponent_beta_x = 3.0
            self.correlation_length_x = 0.3
            self.error_type_x = profiles_simulation.FIGURE_ERROR
            self.rms_x = 0.1
            self.montecarlo_seed_x = 8787
            self.heigth_profile_1D_file_name_x = "mirror_1D_x.dat"
            self.delimiter_x = 0
            self.conversion_factor_x_x = 0.001
            self.conversion_factor_x_y = 1e-6
            self.center_x = 1
            self.modify_x = 0
            self.new_length_x = 0.0
            self.filler_value_x = 0.0
            self.renormalize_x = 0
            self.center_y = 1
            self.modify_y = 0
            self.new_length_y = 0.0
            self.filler_value_y = 0.0
            self.renormalize_y = 1
            self.error_type_y = 0
            self.rms_y = 0.9

def calculate_heigth_profile(input_parameters):
    #### LENGTH
    if input_parameters.kind_of_profile_y == 2:
        combination = "E"

        if input_parameters.delimiter_y == 1:
            profile_1D_y_x, profile_1D_y_y = numpy.loadtxt(input_parameters.heigth_profile_1D_file_name_y, delimiter='\t', unpack=True)
        else:
            profile_1D_y_x, profile_1D_y_y = numpy.loadtxt(input_parameters.heigth_profile_1D_file_name_y, unpack=True)

        profile_1D_y_x *= input_parameters.conversion_factor_y_x
        profile_1D_y_y *= input_parameters.conversion_factor_y_y

        first_coord = profile_1D_y_x[0]
        second_coord = profile_1D_y_x[1]
        last_coord = profile_1D_y_x[-1]
        step = numpy.abs(second_coord - first_coord)
        length = numpy.abs(last_coord - first_coord)
        n_points_old = len(profile_1D_y_x)

        if input_parameters.modify_y == 2:
            profile_1D_y_x_temp = profile_1D_y_x
            profile_1D_y_y_temp = profile_1D_y_y

            if input_parameters.new_length_y > length:
                difference = input_parameters.new_length_y - length

                n_added_points = int(difference / step)
                if difference % step == 0:
                    n_added_points += 1
                if n_added_points % 2 != 0:
                    n_added_points += 1

                profile_1D_y_x = numpy.arange(n_added_points + n_points_old) * step
                profile_1D_y_y = numpy.ones(n_added_points + n_points_old) * input_parameters.filler_value_y * 1e-9 * input_parameters.si_to_user_units
                profile_1D_y_y[int(n_added_points / 2): n_points_old + int(n_added_points / 2)] = profile_1D_y_y_temp
            elif input_parameters.new_length_y < length:
                difference = length - input_parameters.new_length_y

                n_removed_points = int(difference / step)
                if difference % step == 0:
                    n_removed_points -= 1
                if n_removed_points % 2 != 0:
                    n_removed_points -= 1

                if n_removed_points >= 2:
                    profile_1D_y_x = profile_1D_y_x_temp[0: (n_points_old - n_removed_points)]
                    profile_1D_y_y = profile_1D_y_y_temp[(int(n_removed_points / 2) - 1): (n_points_old - int(n_removed_points / 2) - 1)]

                else:
                    profile_1D_y_x = profile_1D_y_x_temp
                    profile_1D_y_y = profile_1D_y_y_temp
            else:
                profile_1D_y_x = profile_1D_y_x_temp
                profile_1D_y_y = profile_1D_y_y_temp

        elif input_parameters.modify_y == 1:
            scale_factor_y = input_parameters.new_length_y / length
            profile_1D_y_x *= scale_factor_y

        if input_parameters.center_y:
            first_coord = profile_1D_y_x[0]
            last_coord = profile_1D_y_x[-1]
            length = numpy.abs(last_coord - first_coord)

            profile_1D_y_x_temp = numpy.linspace(-length / 2, length / 2, len(profile_1D_y_x))
            profile_1D_y_x = profile_1D_y_x_temp

        if input_parameters.renormalize_y == 0:
            rms_y = None
        else:
            if input_parameters.error_type_y == profiles_simulation.FIGURE_ERROR:
                rms_y = input_parameters.rms_y * 1e-9 * input_parameters.si_to_user_units  # from nm to m
            else:
                rms_y = input_parameters.rms_y * 1e-6  # from urad to rad
    else:
        if input_parameters.kind_of_profile_y == 0:
            combination = "F"
        else:
            combination = "G"

        profile_1D_y_x = None
        profile_1D_y_y = None

        if input_parameters.error_type_y == profiles_simulation.FIGURE_ERROR:
            rms_y = input_parameters.rms_y * 1e-9 * input_parameters.si_to_user_units  # from nm to m
        else:
            rms_y = input_parameters.rms_y * 1e-6  # from urad to rad

    #### WIDTH
    if input_parameters.kind_of_profile_x == 2:
        combination += "E"

        if input_parameters.delimiter_x == 1:
            profile_1D_x_x, profile_1D_x_y = numpy.loadtxt(input_parameters.heigth_profile_1D_file_name_x, delimiter='\t', unpack=True)
        else:
            profile_1D_x_x, profile_1D_x_y = numpy.loadtxt(input_parameters.heigth_profile_1D_file_name_x, unpack=True)

        profile_1D_x_x *= input_parameters.conversion_factor_x_x
        profile_1D_x_y *= input_parameters.conversion_factor_x_y

        first_coord = profile_1D_x_x[0]
        second_coord = profile_1D_x_x[1]
        last_coord = profile_1D_x_x[-1]
        step = numpy.abs(second_coord - first_coord)
        length = numpy.abs(last_coord - first_coord)
        n_points_old = len(profile_1D_x_x)

        if input_parameters.modify_x == 2:
            profile_1D_x_x_temp = profile_1D_x_x
            profile_1D_x_y_temp = profile_1D_x_y

            if input_parameters.new_length_x > length:
                difference = input_parameters.new_length_x - length

                n_added_points = int(difference / step)
                if difference % step == 0:
                    n_added_points += 1
                if n_added_points % 2 != 0:
                    n_added_points += 1

                profile_1D_x_x = numpy.arange(n_added_points + n_points_old) * step
                profile_1D_x_y = numpy.ones(n_added_points + n_points_old) * input_parameters.filler_value_x * 1e-9 * input_parameters.si_to_user_units
                profile_1D_x_y[int(n_added_points / 2): n_points_old + int(n_added_points / 2)] = profile_1D_x_y_temp
            elif input_parameters.new_length_x < length:
                difference = length - input_parameters.new_length_x

                n_removed_points = int(difference / step)
                if difference % step == 0:
                    n_removed_points -= 1
                if n_removed_points % 2 != 0:
                    n_removed_points -= 1

                if n_removed_points >= 2:
                    profile_1D_x_x = profile_1D_x_x_temp[0: (n_points_old - n_removed_points)]
                    profile_1D_x_y = profile_1D_x_y_temp[(int(n_removed_points / 2) - 1): (n_points_old - int(n_removed_points / 2) - 1)]

                else:
                    profile_1D_x_x = profile_1D_x_x_temp
                    profile_1D_x_y = profile_1D_x_y_temp
            else:
                profile_1D_x_x = profile_1D_x_x_temp
                profile_1D_x_y = profile_1D_x_y_temp

        elif input_parameters.modify_x == 1:
            scale_factor_x = input_parameters.new_length_x / length
            profile_1D_x_x *= scale_factor_x

        if input_parameters.center_x:
            first_coord = profile_1D_x_x[0]
            last_coord = profile_1D_x_x[-1]
            length = numpy.abs(last_coord - first_coord)

            profile_1D_x_x_temp = numpy.linspace(-length / 2, length / 2, len(profile_1D_x_x))
            profile_1D_x_x = profile_1D_x_x_temp

        if input_parameters.renormalize_x == 0:
            rms_x = None
        else:
            if input_parameters.error_type_x == profiles_simulation.FIGURE_ERROR:
                rms_x = input_parameters.rms_x * 1e-9 * input_parameters.si_to_user_units  # from nm to m
            else:
                rms_x = input_parameters.rms_x * 1e-6  # from urad to rad

    else:
        profile_1D_x_x = None
        profile_1D_x_y = None

        if input_parameters.kind_of_profile_x == 0:
            combination += "F"
        else:
            combination += "G"

        if input_parameters.error_type_x == profiles_simulation.FIGURE_ERROR:
            rms_x = input_parameters.rms_x * 1e-9 * input_parameters.si_to_user_units  # from nm to m
        else:
            rms_x = input_parameters.rms_x * 1e-6  # from urad to rad

    xx, yy, zz = profiles_simulation.simulate_profile_2D(combination=combination,
                                                         mirror_length=input_parameters.dimension_y,
                                                         step_l=input_parameters.step_y,
                                                         random_seed_l=input_parameters.montecarlo_seed_y,
                                                         error_type_l=input_parameters.error_type_y,
                                                         rms_l=rms_y,
                                                         power_law_exponent_beta_l=input_parameters.power_law_exponent_beta_y,
                                                         correlation_length_l=input_parameters.correlation_length_y,
                                                         x_l=profile_1D_y_x,
                                                         y_l=profile_1D_y_y,
                                                         mirror_width=input_parameters.dimension_x,
                                                         step_w=input_parameters.step_x,
                                                         random_seed_w=input_parameters.montecarlo_seed_x,
                                                         error_type_w=input_parameters.error_type_x,
                                                         rms_w=rms_x,
                                                         power_law_exponent_beta_w=input_parameters.power_law_exponent_beta_x,
                                                         correlation_length_w=input_parameters.correlation_length_x,
                                                         x_w=profile_1D_x_x,
                                                         y_w=profile_1D_x_y)

    return xx, yy, zz

def calculate_dabam_profile(input_parameters):
    combination = "E"

    #### LENGTH
    if input_parameters.modify_y == 2:
        profile_1D_y_x_temp = input_parameters.si_to_user_units * input_parameters.server.y
        if input_parameters.use_undetrended == 0:
            profile_1D_y_y_temp = input_parameters.si_to_user_units * input_parameters.server.zHeights
        else:
            profile_1D_y_y_temp = input_parameters.si_to_user_units * input_parameters.server.zHeightsUndetrended

        first_coord = profile_1D_y_x_temp[0]
        second_coord = profile_1D_y_x_temp[1]
        last_coord = profile_1D_y_x_temp[-1]
        step = numpy.abs(second_coord - first_coord)
        length = numpy.abs(last_coord - first_coord)
        n_points_old = len(profile_1D_y_x_temp)

        if input_parameters.new_length_y > length:
            difference = input_parameters.new_length_y - length

            n_added_points = int(difference / step)
            if difference % step == 0:
                n_added_points += 1
            if n_added_points % 2 != 0:
                n_added_points += 1

            profile_1D_y_x = numpy.arange(n_added_points + n_points_old) * step
            profile_1D_y_y = numpy.ones(n_added_points + n_points_old) * input_parameters.filler_value_y * 1e-9 * input_parameters.si_to_user_units
            profile_1D_y_y[int(n_added_points / 2): n_points_old + int(n_added_points / 2)] = profile_1D_y_y_temp
        elif input_parameters.new_length_y < length:
            difference = length - input_parameters.new_length_y

            n_removed_points = int(difference / step)
            if difference % step == 0:
                n_removed_points -= 1
            if n_removed_points % 2 != 0:
                n_removed_points -= 1

            if n_removed_points >= 2:
                profile_1D_y_x = profile_1D_y_x_temp[0: (n_points_old - n_removed_points)]
                profile_1D_y_y = profile_1D_y_y_temp[(int(n_removed_points / 2) - 1): (n_points_old - int(n_removed_points / 2) - 1)]

            else:
                profile_1D_y_x = profile_1D_y_x_temp
                profile_1D_y_y = profile_1D_y_y_temp
        else:
            profile_1D_y_x = profile_1D_y_x_temp
            profile_1D_y_y = profile_1D_y_y_temp

    else:
        if input_parameters.modify_y == 0:
            profile_1D_y_x = input_parameters.si_to_user_units * input_parameters.server.y
        elif input_parameters.modify_y == 1:
            scale_factor_y = input_parameters.new_length_y / (input_parameters.si_to_user_units * (max(input_parameters.server.y) - min(input_parameters.server.y)))
            profile_1D_y_x = input_parameters.si_to_user_units * input_parameters.server.y * scale_factor_y

        if input_parameters.use_undetrended == 0:
            profile_1D_y_y = input_parameters.si_to_user_units * input_parameters.server.zHeights
        else:
            profile_1D_y_y = input_parameters.si_to_user_units * input_parameters.server.zHeightsUndetrended

    if input_parameters.center_y:
        first_coord = profile_1D_y_x[0]
        last_coord = profile_1D_y_x[-1]
        length = numpy.abs(last_coord - first_coord)

        profile_1D_y_x_temp = numpy.linspace(-length / 2, length / 2, len(profile_1D_y_x))

        profile_1D_y_x = profile_1D_y_x_temp

    if input_parameters.renormalize_y == 0:
        rms_y = None
    else:
        if input_parameters.error_type_y == profiles_simulation.FIGURE_ERROR:
            rms_y = input_parameters.rms_y * 1e-9 * input_parameters.si_to_user_units  # from nm to m
        else:
            rms_y = input_parameters.rms_y * 1e-6  # from urad to rad

    #### WIDTH
    if input_parameters.kind_of_profile_x == 3:
        combination += "F"

        xx, yy, zz = profiles_simulation.simulate_profile_2D(combination=combination,
                                                             error_type_l=input_parameters.error_type_y,
                                                             rms_l=rms_y,
                                                             x_l=profile_1D_y_x,
                                                             y_l=profile_1D_y_y,
                                                             mirror_width=input_parameters.dimension_x,
                                                             step_w=input_parameters.step_x,
                                                             rms_w=0.0)
    else:
        if input_parameters.kind_of_profile_x == 2:
            combination += "E"

            if input_parameters.delimiter_x == 1:
                profile_1D_x_x, profile_1D_x_y = numpy.loadtxt(input_parameters.heigth_profile_1D_file_name_x, delimiter='\t', unpack=True)
            else:
                profile_1D_x_x, profile_1D_x_y = numpy.loadtxt(input_parameters.heigth_profile_1D_file_name_x, unpack=True)

            profile_1D_x_x *= input_parameters.conversion_factor_x_x
            profile_1D_x_y *= input_parameters.conversion_factor_x_y

            first_coord = profile_1D_x_x[0]
            second_coord = profile_1D_x_x[1]
            last_coord = profile_1D_x_x[-1]
            step = numpy.abs(second_coord - first_coord)
            length = numpy.abs(last_coord - first_coord)
            n_points_old = len(profile_1D_x_x)

            if input_parameters.modify_x == 2:
                profile_1D_x_x_temp = profile_1D_x_x
                profile_1D_x_y_temp = profile_1D_x_y

                if input_parameters.new_length_x > length:
                    difference = input_parameters.new_length_x - length

                    n_added_points = int(difference / step)
                    if difference % step == 0:
                        n_added_points += 1
                    if n_added_points % 2 != 0:
                        n_added_points += 1

                    profile_1D_x_x = numpy.arange(n_added_points + n_points_old) * step
                    profile_1D_x_y = numpy.ones(n_added_points + n_points_old) * input_parameters.filler_value_x * 1e-9 * input_parameters.si_to_user_units
                    profile_1D_x_y[int(n_added_points / 2): n_points_old + int(n_added_points / 2)] = profile_1D_x_y_temp
                elif input_parameters.new_length_x < length:
                    difference = length - input_parameters.new_length_x

                    n_removed_points = int(difference / step)
                    if difference % step == 0:
                        n_removed_points -= 1
                    if n_removed_points % 2 != 0:
                        n_removed_points -= 1

                    if n_removed_points >= 2:
                        profile_1D_x_x = profile_1D_x_x_temp[0: (n_points_old - n_removed_points)]
                        profile_1D_x_y = profile_1D_x_y_temp[(int(n_removed_points / 2) - 1): (n_points_old - int(n_removed_points / 2) - 1)]

                    else:
                        profile_1D_x_x = profile_1D_x_x_temp
                        profile_1D_x_y = profile_1D_x_y_temp
                else:
                    profile_1D_x_x = profile_1D_x_x_temp
                    profile_1D_x_y = profile_1D_x_y_temp

            elif input_parameters.modify_x == 1:
                scale_factor_x = input_parameters.new_length_x / length
                profile_1D_x_x *= scale_factor_x

            if input_parameters.center_x:
                first_coord = profile_1D_x_x[0]
                last_coord = profile_1D_x_x[-1]
                length = numpy.abs(last_coord - first_coord)

                profile_1D_x_x_temp = numpy.linspace(-length / 2, length / 2, len(profile_1D_x_x))
                profile_1D_x_x = profile_1D_x_x_temp

            if input_parameters.renormalize_x == 0:
                rms_x = None
            else:
                if input_parameters.error_type_x == profiles_simulation.FIGURE_ERROR:
                    rms_x = input_parameters.rms_x * 1e-9 * input_parameters.si_to_user_units  # from nm to m
                else:
                    rms_x = input_parameters.rms_x * 1e-6  # from urad to rad

        else:
            profile_1D_x_x = None
            profile_1D_x_y = None

            if input_parameters.kind_of_profile_x == 0:
                combination += "F"
            else:
                combination += "G"

            if input_parameters.error_type_x == profiles_simulation.FIGURE_ERROR:
                rms_x = input_parameters.rms_x * 1e-9 * input_parameters.si_to_user_units  # from nm to m
            else:
                rms_x = input_parameters.rms_x * 1e-6  # from urad to rad

        xx, yy, zz = profiles_simulation.simulate_profile_2D(combination=combination,
                                                             error_type_l=input_parameters.error_type_y,
                                                             rms_l=rms_y,
                                                             x_l=profile_1D_y_x,
                                                             y_l=profile_1D_y_y,
                                                             mirror_width=input_parameters.dimension_x,
                                                             step_w=input_parameters.step_x,
                                                             random_seed_w=input_parameters.montecarlo_seed_x,
                                                             error_type_w=input_parameters.error_type_x,
                                                             rms_w=rms_x,
                                                             power_law_exponent_beta_w=input_parameters.power_law_exponent_beta_x,
                                                             correlation_length_w=input_parameters.correlation_length_x,
                                                             x_w=profile_1D_x_x,
                                                             y_w=profile_1D_x_y)

    return xx, yy, zz

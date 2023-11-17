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

class BenderStructuralParameters:
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
                 workspace_units_to_m=None,
                 workspace_units_to_mm=None):
        self.dim_x_minus = dim_x_minus 
        self.dim_x_plus = dim_x_plus 
        self.bender_bin_x = bender_bin_x 
        self.dim_y_minus = dim_y_minus 
        self.dim_y_plus = dim_y_plus 
        self.bender_bin_y = bender_bin_y
        self.p                = p
        self.q                = q
        self.grazing_angle    = grazing_angle
        self.E = E
        self.h = h 
        self.figure_error_mesh = figure_error_mesh
        self.workspace_units_to_m = workspace_units_to_m
        self.workspace_units_to_mm = workspace_units_to_mm

# ----------------------------------------------------------------------
# Bender FIT
# ----------------------------------------------------------------------

class BenderFitParameters():
    def __init__(self,
                 optimized_length = None,
                 n_fit_steps=None):
        self.optimized_length = optimized_length
        self.n_fit_steps      = n_fit_steps

class BenderOuputData:
    def __init__(self,
                 x=None,
                 y=None,
                 ideal_profile=None,
                 bender_profile=None,
                 correction_profile=None,
                 titles=None,
                 z_bender_correction=None,
                 z_figure_error=None,
                 z_bender_correction_no_figure_error=None):
        self.x = x
        self.y = y
        self.ideal_profile       = ideal_profile
        self.bender_profile      = bender_profile
        self.correction_profile  = correction_profile
        self.titles              = titles
        self.z_bender_correction = z_bender_correction
        self.z_figure_error      = z_figure_error
        self.z_bender_correction_no_figure_error=z_bender_correction_no_figure_error

# ----------------------------------------------------------------------
# Bender Movements
# ----------------------------------------------------------------------

class BenderMovement():
    def __init__(self, position_upstream=None, position_downstream=None):
        self.__position_upstream   = position_upstream
        self.__position_downstream = position_downstream

    @property
    def position_upstream(self): return self.__position_upstream
    @property
    def position_downstream(self): return self.__position_downstream
"""
This is a collection of functions to simulate profiles that can be used for describing surface errors in optical surfaces


Note that all the functions are dimensionless: therefore use always the same unit in horizontal, vertical, and RMS inputs.
    Angles are in rad.

Functions:
    combine_two_transversal_profiles(): combine two profiles into a mesh
    simulate_gaussian_profile_1D():
    simulate_fractal_profile_1D():
    simulate_profile_2D
    simulate_profile_2D_from_1D
    create_random_rough_surface_1D(): binding to simulate_gaussian_profile_1D and simulate_fractal_profile_1D

    create_simulated_1D_file_APS
    create_simulated_2D_profile_APS
    create_2D_profile_from_1D

Tests utils:
    package_dirname: gets the package path to retrieve test file
    slopes: calculate the slope errors

Tests:
    test_1d_gaussian
    test_1d_fractal
    test_1d
    test_2d_new
    test_2d
    test_2d_from_1d_new
    test_2d_from_1d

Authors and main contributors:
    Luca Rebuffi, Ruben Reininger, Manuel Sanchez del Rio, Xianbo Shi

"""

import numpy
#todo: rename file to simulate_profiles

#todo: remove these global variables?
# define either profile_type=0, or better place a string in the flag, e.g.: profile_type='gaussian'

FIGURE_ERROR = 0
SLOPE_ERROR = 1

GAUSSIAN = 0
FRACTAL = 1


#########################################################
#
# these are the new routines
#
#########################################################

# "binding" for GAUSSIAN or FRACTAL
def simulate_profile_1D(step=1.0, mirror_length=200.0, random_seed=8787,
                        error_type=FIGURE_ERROR, rms=3e-9,
                        profile_type=GAUSSIAN,
                        rms_heights=3e-9,              # specific inputs for profile_type=GAUSSIAN,
                        correlation_length=30.,        # specific inputs for profile_type=GAUSSIAN,
                        power_law_exponent_beta=0.9,   # specific inputs for profile_type=FRACTAL,
                        ):
    """
    binding to simulate_profile_1D_gaussian and simulate_profile_1D_fractal
    :param step:
    :param mirror_length:
    :param random_seed:
    :param error_type:
    :param rms:
    :param profile_type:
    :param rms_heights:
    :param correlation_length:
    :param power_law_exponent_beta:
    :return:
    """
    if error_type == FIGURE_ERROR:
        renormalize_to_heights_sd = rms
        renormalize_to_slopes_sd = None
    elif error_type == SLOPE_ERROR:
        renormalize_to_heights_sd = None
        renormalize_to_slopes_sd = rms
    else:
        raise Exception("simulate_profile_1D: wrong error_type")

    if profile_type==GAUSSIAN:
        x_coords, y_values = simulate_profile_1D_gaussian(step=step, \
                                    mirror_length=mirror_length,\
                                    random_seed=random_seed,
                                    rms_heights=rms_heights,\
                                    correlation_length=correlation_length,\
                                    renormalize_to_heights_sd=renormalize_to_heights_sd, \
                                    renormalize_to_slopes_sd=renormalize_to_slopes_sd,\
                                    )
    elif profile_type==FRACTAL:
        x_coords, y_values = simulate_profile_1D_fractal(step=step,\
                                    mirror_length=mirror_length,\
                                    random_seed=random_seed,
                                    power_law_exponent_beta=power_law_exponent_beta,\
                                    renormalize_to_heights_sd=renormalize_to_heights_sd, \
                                    renormalize_to_slopes_sd=renormalize_to_slopes_sd,\
                                    )
    else:
        raise Exception("simulate_profile_1D: Profile type not recognized")

    return x_coords, y_values


def simulate_profile_1D_gaussian(step=1.0, npoints=None, mirror_length=200.0, rms_heights=3e-9, correlation_length=30.0,
                                 random_seed=8787,renormalize_to_heights_sd=None,renormalize_to_slopes_sd=None):
    """
    #
    # this is based on a translation to python of the matlab function rsgeng1d from David Bergström
    # see: http://www.mysimlabs.com/surface_generation.html
    #
    # % generates a 1-dimensional random rough surface f(x) with N surface points.
    # % The surface has a Gaussian height distribution function and a Gaussian
    # % autocovariance function, where rL is the length of the surface, h is the
    # % RMS height and cl is the correlation length.
    #
    :param step: step in mirror length (default=0.2)
    :param mirror_length: profile length
    :param npoints: number of points in mirror length (default=None, i.e., undefined so use step and mirror_length
                    to calculate it. If defined, use npoints and step is irrelevant)
    :param rms_heights: rms height for the Gaussian. It is usually not important if normalize_to_{heights,slopes_sd is used}
    :param correlation_length: correlation length
    :param random_seed: a random seed to initialize numpy.seed(). Use zero to avoid initialization (default=8787)
    :param renormalize_to_heights_sd: set to a value to renormalize the profile to this height stdev value (default=None)
    :param renormalize_to_slopes_sd: set to a value to renormalize the profile to this slope stdev value (default=None)
    :return: (x,f) where x = profile abscissas, f = profile heights
    """
    #
    # this is a translation to python of the matlab function rsgeng1d from David Bergström
    # see: http://www.mysimlabs.com/surface_generation.html
    #---------------
    # function [f,x] = rsgeng1D(N,rL,h,cl)
    # %
    # % [f,x] = rsgeng1D(N,rL,h,cl)
    # %
    # % generates a 1-dimensional random rough surface f(x) with N surface points.
    # % The surface has a Gaussian height distribution function and a Gaussian
    # % autocovariance function, where rL is the length of the surface, h is the
    # % RMS height and cl is the correlation length.
    # %
    # % Input:    N   - number of surface points
    # %           rL  - length of surface
    # %           h   - rms height
    # %           cl  - correlation length
    # %
    # % Output:   f   - surface heights
    # %           x   - surface points
    # %
    # % Last updated: 2010-07-26 (David Bergström).
    # %
    #---------------

    if npoints is None:
        n_surface_points = int(1 + (mirror_length / step))
    else:
        n_surface_points = npoints

    if random_seed != 0:
        numpy.random.seed(seed=random_seed)

    # format long;
    #
    # x = linspace(-rL/2,rL/2,N);
    x_coords = numpy.linspace(-mirror_length / 2, mirror_length / 2, n_surface_points)

    #
    # Z = h.*randn(1,N); % uncorrelated Gaussian random rough surface distribution
    #                      % with mean 0 and standard deviation h
    #
    uncorrelated_gaussian_random_rough_surface = rms_heights * numpy.random.randn(1, int(n_surface_points))
    uncorrelated_gaussian_random_rough_surface.shape = -1

    # % Gaussian filter
    # F = exp(-x.^2/(cl^2/2));
    gaussian_filter = numpy.exp(-x_coords**2 / (correlation_length ** 2 / 2))
    #
    # % correlation of surface using convolution (faltung), inverse
    # % Fourier transform and normalizing prefactors
    # f = sqrt(2/sqrt(pi))*sqrt(rL/N/cl)*ifft(fft(Z).*fft(F));
    y_values = numpy.sqrt(2/numpy.sqrt(numpy.pi))*numpy.sqrt(mirror_length / n_surface_points / correlation_length) * \
               numpy.fft.ifft(numpy.fft.fft(uncorrelated_gaussian_random_rough_surface) * numpy.fft.fft(gaussian_filter))

    #added srio@esrf.eu
    #Although the created profile should have a SD close to rms_heights, it depends on statistics.
    #therefore, it is better to renormaliza the profile to have the exact SD value for height
    #it also permits to renormalize the profile to have the wanted slope error
    if renormalize_to_heights_sd != None:
        y_values = y_values / y_values.std() * renormalize_to_heights_sd

    if renormalize_to_slopes_sd != None:
        yslopes = numpy.gradient(y_values, x_coords[1]-x_coords[0])
        y_values = y_values / yslopes.std() * renormalize_to_slopes_sd
    return x_coords, y_values


def simulate_profile_1D_fractal(step=1.0, npoints=None, mirror_length=200.0,
                                power_law_exponent_beta=1.5,npoints_ratio_f_over_x=1.0, random_seed=8787,
                                renormalize_to_heights_sd=None,renormalize_to_slopes_sd=None,
                                frequency_max=None,frequency_min=None):
    """
    #
    # generates a 1-dimensional random rough surface z(x) with n_surface_points surface points.
    # The surface has a power lar PSD |f|**(-beta).
    # It is a fractal profile if 1<beta<3
    #
    :param step: step in mirror length (default=0.2)
    :param mirror_length: profile length
    :param npoints: number of points in mirror length (default=None, i.e., undefined so use step and mirror_length
                    to calculate it. If defined, use npoints and step is irrelevant)
    :param power_law_exponent_beta: beta value
    :param npoints_ratio_f_over_x: ratio of the number of points in frequency domain over real space (default=1.0)
    :param random_seed: a random seed to initialize numpy.seed(). Use zero to avoid initialization (default=8787)
    :param renormalize_to_heights_sd: set to a value to renormalize the profile to this height stdev value (default=None)
    :param renormalize_to_slopes_sd: set to a value to renormalize the profile to this slope stdev value (default=None)
    :param frequency_max:
    :param frequency_min:
    :return: (x,prof) where x = profile abscissas, prof = profile heights
    """

    if npoints is None:
        n_surface_points = int(1 + (mirror_length / step))
    else:
        n_surface_points = npoints

    if random_seed != 0:
        numpy.random.seed(seed=random_seed)


    x_coords = numpy.linspace(-0.5*mirror_length,0.5*mirror_length,n_surface_points)


    if frequency_min is None:
        f_from =  1/(1*mirror_length)
    else:
        f_from = frequency_min

    if frequency_max is None:
        f_to = 1/(2*step)
    else:
        f_to = frequency_max

    if npoints_ratio_f_over_x == 1.0:
        f_npoints = n_surface_points
    else:
        f_npoints = int(n_surface_points*npoints_ratio_f_over_x)

    freq = numpy.linspace(f_from,f_to,f_npoints)
    #todo: make exponent of power law a parameter
    ampl = freq**(-power_law_exponent_beta/2)
    phases = numpy.random.rand(freq.size)*2*numpy.pi
    ymirr = numpy.zeros(n_surface_points)
    for i in range(f_npoints):
        ymirr += (ampl[i] *  numpy.sin(2*numpy.pi*freq[i]*x_coords + phases[i]))

    if renormalize_to_heights_sd != None:
        ymirr = ymirr / ymirr.std() * renormalize_to_heights_sd

    if renormalize_to_slopes_sd != None:
        yslopes = numpy.gradient(ymirr, step)
        ymirr = ymirr / yslopes.std() * renormalize_to_slopes_sd

    return x_coords, ymirr

# Combines two 1D simulated (GAUSSIAN or FRACTAL) or EXPERIMENTAL simulated profiles into a single 2D profile or surface
def simulate_profile_2D(combination='FF',
                                mirror_length=200.0, step_l=1.0, random_seed_l=8787, error_type_l=FIGURE_ERROR, rms_l=1e-6,
                                power_law_exponent_beta_l=1.5,correlation_length_l=30.0,x_l=None, y_l=None,
                                mirror_width=20.000, step_w=1.0, random_seed_w=8788, error_type_w=FIGURE_ERROR, rms_w=1e-6,
                                power_law_exponent_beta_w=1.5,correlation_length_w=30.0,x_w=None, y_w=None, ):
    """
    Combines two 1D simulated (GAUSSIAN or FRACTAL) or EXPERIMENTAL simulated profiles into a single 2D profile or surface

    :param combination: two character string with the comination of profile type (F-fractal, G=gaussian, E=experimental)
                        The first character is for mirror length direction (Y, subscript _l) and
                        the second character is for mirror width direction (X, subscript _w)
                        Example: "FF" (fractal in Y, fractal in X), "EG" (Experimental in Y, Gaussian in X"

    :param mirror_length: the mirror length (along Y)
    :param step_l: step size
    :param random_seed_l: seed to initialize random seed for Y simulation
    :param error_type_l: normalize to heights error (0) or slopes error (1)
    :param rms_l:  the vealue of eithe height error (if error_type_l=0) or slope error (if error_type_l=1)
    :param power_law_exponent_beta_l: if Fractal, the beta value
    :param correlation_length_l:  if Gaussian, the correlation length
    :param x_l: if Experimental, the abscissas (Y coordinales)
    :param y_l: if Experimental, the ordinates (heights)
    The following parameters are the same as before, but for the direction along the width (X)
    :param mirror_width:
    :param step_w:
    :param random_seed_w:
    :param error_type_w:
    :param rms_w:
    :param power_law_exponent_beta_w:
    :param correlation_length_w:
    :param x_w:
    :param y_w:

    :return: WW_x, SF_x, s  arrays along width (X), length (Y) and heights(SF_x.size,WW_x.size)
    """
    if error_type_l == FIGURE_ERROR:
        renormalize_to_heights_sd_l = rms_l
        renormalize_to_slopes_sd_l = None
    else:
        renormalize_to_heights_sd_l = None
        renormalize_to_slopes_sd_l = rms_l

    if error_type_w == FIGURE_ERROR:
        renormalize_to_heights_sd_w = rms_w
        renormalize_to_slopes_sd_w = None
    else:
        renormalize_to_heights_sd_w = None
        renormalize_to_slopes_sd_w = rms_w

    #
    # compute profile along mirror length
    #
    if combination[0] == "F":
        SF_x, SF = simulate_profile_1D_fractal(step=step_l, \
                                           mirror_length=mirror_length, \
                                           power_law_exponent_beta=power_law_exponent_beta_l, \
                                           random_seed=random_seed_l, \
                                           renormalize_to_heights_sd=renormalize_to_heights_sd_l,\
                                           renormalize_to_slopes_sd=renormalize_to_slopes_sd_l)
    elif combination[0] == "G":
        SF_x, SF = simulate_profile_1D_gaussian(step=step_l, \
                                           mirror_length=mirror_length, \
                                           correlation_length=correlation_length_l, rms_heights = rms_l, \
                                           random_seed=random_seed_l, \
                                           renormalize_to_heights_sd=renormalize_to_heights_sd_l,\
                                           renormalize_to_slopes_sd=renormalize_to_slopes_sd_l)
    elif combination[0] == "E":
        if (x_l is None or y_l is None):
            raise Exception("simulate_profile_2D: no input arrays found: x_l, y_l")
        else:
            SF_x, SF = x_l, y_l

            if renormalize_to_heights_sd_l != None:
                SF = SF / SF.std() * renormalize_to_heights_sd_l

            if renormalize_to_slopes_sd_l != None:
                yslopes = numpy.gradient(SF, SF_x[1]-SF_x[0])
                SF = SF / yslopes.std() * renormalize_to_slopes_sd_l
    else:
        raise Exception("simulate_profile_2D: illegal combination code")

    #
    # compute profile along mirror width
    #
    if combination[1] == "F":
        WW_x, WW = simulate_profile_1D_fractal(step=step_w, \
                                           mirror_length=mirror_width, \
                                           power_law_exponent_beta=power_law_exponent_beta_w, \
                                           random_seed=random_seed_w, \
                                           renormalize_to_heights_sd=renormalize_to_heights_sd_w,\
                                           renormalize_to_slopes_sd=renormalize_to_slopes_sd_w)
    elif combination[1] == "G":
        WW_x, WW = simulate_profile_1D_gaussian(step=step_w, \
                                           mirror_length=mirror_width, \
                                           correlation_length=correlation_length_w, rms_heights=rms_w, \
                                           random_seed=random_seed_w, \
                                           renormalize_to_heights_sd=renormalize_to_heights_sd_w,\
                                           renormalize_to_slopes_sd=renormalize_to_slopes_sd_w)
    elif combination[1] == "E":
        if (x_w is None or y_w is None):
            raise Exception("simulate_profile_2D: no input arrays found: x_w, y_w")
        else:
            WW_x, WW = x_w, y_w

            if renormalize_to_heights_sd_w != None:
                WW = WW / WW.std() * renormalize_to_heights_sd_w

            if renormalize_to_slopes_sd_w != None:
                yslopes = numpy.gradient(WW, WW_x[1]-WW_x[0])
                WW = WW / yslopes.std() * renormalize_to_slopes_sd_w


    else:
        raise Exception("simulate_profile_2D: illegal combination code")

    s = combine_two_transversal_profiles(WW_x, WW, SF_x, SF)

    slp = slopes(s.T,WW_x,SF_x,silent=1, return_only_rms=1)

    #
    # now readjust normalization to match the longitudinal profile
    #
    if error_type_l == FIGURE_ERROR:
        if not rms_l is None and rms_l != 0.0:
            s *= rms_l / s.std()
    else:
        slp = slopes(s.T,WW_x,SF_x,silent=1, return_only_rms=1)
        if not rms_l is None and rms_l != 0.0:
            s *= rms_l / slp[1]
        else:
            if not rms_w is None and rms_w != 0:
                pass
                #s *= rms_w / slp[0]

    return WW_x, SF_x, s


#########################################################
#
# These are the routines translated from APS igor code. TODO: remove and replace by the new ones?
#
#########################################################

def create_simulated_1D_file_APS(mirror_length=200.0, step=1.0, random_seed=8787, error_type=FIGURE_ERROR, rms=1e-6,
                                 power_law_exponent_beta=1.5, power_law_exponent_beta_two=1.5,
                                 frequency_power_law_match=0.001):

    """
    #
    # generates a 1-dimensional random rough surface z(x) with n_surface_points surface points.
    # The surface has a power law PSD |f|**(-beta)
    # where beta=power_law_exponent_beta for frequencies < frequency_power_law_match
    #       beta=power_law_exponent_beta for frequencies > frequency_power_law_match
    #
    # It is a fractal profile if 1<beta<3
    #

    :param mirror_length: the mirror length (mm or any user unit) (default=200.0)
    :param step: the mirror step (mm or user units) (default=1.0)
    :param random_seed: a random seed to initialize numpy.seed(). Use zero to avoid initialization (default=8787)
    :param error_type: define wherer normalize the output profile to height error (0, default) or slope error (1)
    :param rms:  either the heigh error in user units (if error_type=0) or the slope error in rad (error_type=1) (default=1e-6)
    :param power_law_exponent_beta: the beta value of the first interval of frequencies (default=1.5)
    :param power_law_exponent_beta_two: the beta value of the second interval of frequencies (default=1.5)
    :param frequency_power_law_match: the frequency (in 1/(user units) to match frequency intervals (default=0.001)
    :return:

    :param mirror_length: "Enter mirror length, even"
    :param step: "Step for length "
    :param random_seed: initialize numpy random seed to this value. Set to zero to avoid initialization
    :param error_type: "Figure error (0) or Slope error (1)"
    :param rms: "RMS value of the above"
    :return: x coords, y values
    """
    # if(step ==0):
    #     mirror_length=200.0	#Number of points surface wave
    #     step=1			      #Spacing surface wave
    #     random_seed=8787
    #     error_type=FIGURE_ERROR
    #     rms=0.1e-6

    if random_seed != 0:
        numpy.random.seed(seed=random_seed)


    mult1 = 2.1e-10  # a change in this value does not alter results, as profile is changet to match rms
    mult2 = mult1    # todo: the ratio multi2/multi1 can be an external parameter
    slo1 = -power_law_exponent_beta
    slo2 = -power_law_exponent_beta_two
    chSlo = frequency_power_law_match

    npo=int(mirror_length / step + 1)

    error_profile_x = numpy.linspace(-mirror_length / 2, mirror_length / 2, npo)
    error_profile = numpy.zeros(npo)

    freq= 1.0 / mirror_length

    #todo: the frequency array goes from freq to freq+1~1, Why? so max frequency corresponds roughly to 1mm
    x = numpy.linspace(freq, freq + freq * ((npo-1) * step), npo)
    FouAmp = numpy.zeros(npo)
    FouPha = numpy.zeros(npo)
    FouFre = x

    for i in range(npo):
        if (FouFre[i] < chSlo):
            FouAmp[i]=mult1*FouFre[i]**slo1
        else:
            FouAmp[i]=mult2*FouFre[i]**slo2

    for i in range(npo):
        #FouPha[i] = enoise(numpy.pi)
        FouPha[i] = numpy.pi * numpy.random.rand()
        error_profile += FouAmp[i]*numpy.cos(-numpy.pi*2*error_profile_x*FouFre[i]+FouPha[i])
        #todo: prefer this one?:
        #error_profile += FouAmp[i]*numpy.sin(numpy.pi*2*error_profile_x*FouFre[i]+2*FouPha[i])
        # shift profile to go trought origin (0,0) ?

    if (error_type == SLOPE_ERROR): # :TODO check this passage!!!!!
        SF_DIF = numpy.gradient(error_profile, step)
        V_sdev = SF_DIF.std()
        error_profile *= rms / V_sdev
    elif error_type == FIGURE_ERROR:
        V_sdev = error_profile.std()
        error_profile *= rms / V_sdev

    return error_profile_x, error_profile

def create_simulated_2D_profile_APS(mirror_length=200.0, step_l=1.0, random_seed_l=8787, error_type_l=FIGURE_ERROR, rms_l=1e-6,
                                    mirror_width=20.000, step_w=1.0, random_seed_w=8788, error_type_w=FIGURE_ERROR, rms_w=1e-6,
                                    power_law_exponent_beta_l=1.5,power_law_exponent_beta_w=1.5):
    """
    #
    # generates a 2-dimensional random rough surface z(x,y) with PSDs following a power law
    # The surface has a power law PSD |f|**(-beta) in both y and x directions
    #
    # It is a fractal profile if 1<beta<3
    :param  mirror_length: the mirror length (mm or any user unit) (default=200.0)
    :param  mirror_with: the mirror width (mm or any user unit) (default=200.0)
    :param  step_l (step_w): the step for mirror length (width) (mm or user units) (default=1.0)
    :param  random_seed_l (random_seed_r): a random seed to initialize numpy.seed() when creating the longitudinal (transversal) profiles
    :param  error_type_l (_w): normalize the output profile to height error (0, default) or slope error (1)
    :param  rms_l (_w):  either the heigh error in user units (if error_type_{l,w}=0) or the slope error in rad (error_type=1)
    :param  power_law_exponent_beta: the beta value of the first interval of frequencies (default=1.5)
    :param  power_law_exponent_beta_l (_w): the beta value for the longitudinal (transversal) profile


    :param error_type_w: "Figure error (0) or Slope error (1)"
    :param rms_w: "RMS value of the above"
    :return: x, y, z arrays for width direction x, longitudinal direction y, and heights z(x,y)
    """

    WW_x, WW = create_simulated_1D_file_APS(mirror_width, step_w, random_seed_w, error_type_w, rms_w, \
                                            power_law_exponent_beta=power_law_exponent_beta_l)
    SF_x, SF = create_simulated_1D_file_APS(mirror_length, step_l, random_seed_l, error_type_l, rms_l, \
                                            power_law_exponent_beta_two=power_law_exponent_beta_w)

    s = combine_two_transversal_profiles(WW_x, WW, SF_x, SF)

    return WW_x, SF_x, s

def create_2D_profile_from_1D(profile_1D_x, profile_1D_y, mirror_width=20.0, step_w=1.0, random_seed_w=8787,
                              error_type_w=FIGURE_ERROR, rms_w=1e-6):
    """
    #
    # generates a 2-dimensional random rough surface z(x,y) with PSD following a power law
    # The surface has a power law PSD |f|**(-beta) in both y and x directions
    #
    # It is a fractal profile if 1<beta<3
    :param profile_1D_x:
    :param profile_1D_y:
    :param mirror_width:
    :param step_w:
    :param random_seed_w:
    :param error_type_w:
    :param rms_w:
    :return:
    """

    WW_x, WW = create_simulated_1D_file_APS(mirror_width, step_w, random_seed_w, error_type_w, rms_w)
    SF_x, SF = profile_1D_x, profile_1D_y

    s = combine_two_transversal_profiles(WW_x, WW, SF_x, SF)

    return WW_x, SF_x, s

#########################################################
#
# TOOLS
#
#########################################################

#TODO: note that the returned array las LENGTH in zeroth order and WIDTH in first order, so shadow coordinates (Y,X)
def combine_two_transversal_profiles(WW_x, WW, SF_x, SF):
    """
    combine two profiles into a mesh
    :param WW_x: abscissas of profile along width
    :param WW: profile along width
    :param SF_x: abscissas of profile along length
    :param SF: profile along length
    :return: the combined mesh s(index_length,index_width)
    """
    npoL = SF.size
    npoW = WW.size

    s = numpy.zeros((npoL, npoW))
    for i in range(npoW):
        s[:,i] = SF + WW[i]

    return s


#copied from ShadowTools,py
def slopes(z,x,y,silent=1, return_only_rms=0):
    """
    ;+
    ; NAME:
    ;	slopes
    ; PURPOSE:
    ;       This function calculates the slope errors of a surface along the mirror
    ;       length y and mirror width x.
    ; CATEGORY:
    ;	SHADOW tools
    ; CALLING SEQUENCE:
    ;	(slope,slopesrms) = slopes(z,x,y)
    ; INPUTS:
    ;	x: the width array of dimensions (Nx)
    ;	y: the length array of dimensions (Ny)
    ;	z: the surface array of dimensions (Nx,Ny)
    ; OUTPUTS:
    ;   slope: an array of dimension (2,Nx,Ny) with the slopes errors in rad
    ;            along X in out[0,:,:] and along Y in out[1,:,:]
    ;	slopesrms: a 2-dim array with (in rad): [slopeErrorRMS_X,slopeErrorRMS_Y]
    ;
    ; MODIFICATION HISTORY:
    ;       MSR 1994 written
    ;       2016-02-17 luca.rebuffi@elettra.eu modified calculation of nx,ny
    ;       2015-04-08 srio@esrf.eu makes calculations in double precision.
    ;       2014-09-11 documented
    ;       2012-02-10 srio@esrf.eu python version
    ;-
    ;
    """

    nx = x.size
    ny = y.size

    slope = numpy.zeros((2,nx,ny))

    #;
    #; slopes in x direction
    #;
    for i in range(nx-1):
        step = x[i+1] - x[i]
        slope[0,i,:] = numpy.arctan( (z[i+1,:] - z[i,:] ) / step )
    slope[0,nx-1,:] = slope[0,nx-2,:]

    #;
    #; slopes in y direction
    #;
    for i in range(ny-1):
        step = y[i+1] - y[i]
        slope[1,:,i] = numpy.arctan( (z[:,i+1] - z[:,i] ) / step )
    slope[1,:,ny-1] = slope[1,:,ny-2]

    slopermsX = slope[0,:,:].std()
    slopermsY = slope[1,:,:].std()
    slopermsXsec = slopermsX*180.0/numpy.pi*3600.0
    slopermsYsec = slopermsY*180.0/numpy.pi*3600.0
    # srio changed to dimensionless:
    # slopesrms = numpy.array([slopermsXsec,slopermsYsec, slopermsX*1e6,slopermsY*1e6])
    slopesrms = numpy.array([slopermsX,slopermsY])

    if not(silent):
        print('\n **** slopes: ****')
        print(' Slope error rms in X direction: %f arcsec'%(slopermsXsec))
        print('                               : %f urad'%(slopermsX*1e6))
        print(' Slope error rms in Y direction: %f arcsec'%(slopermsYsec))
        print('                               : %f urad'%(slopermsY*1e6))
        print(' *****************')

    if return_only_rms:
        return slopesrms
    else:
        return (slope,slopesrms)




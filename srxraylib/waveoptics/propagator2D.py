import numpy


from srxraylib.util.data_structures import ScaledMatrix
from srxraylib.waveoptics.wavefront2D import Wavefront2D

# TODO: check resulting amplitude normalization

def propagate_2D_fraunhofer(wavefront, propagation_distance=1.0):
    """
    2D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :return: a new 2D wavefront object with propagated wavefront
    """
    wavelength = wavefront.get_wavelength()

    #
    # check validity
    #
    x =  wavefront.get_coordinate_x()
    y =  wavefront.get_coordinate_y()
    half_max_aperture = 0.5 * numpy.array( (x[-1]-x[0],y[-1]-y[0])).max()
    far_field_distance = half_max_aperture**2/wavelength
    if propagation_distance < far_field_distance:
        print("WARNING: Fraunhoffer diffraction valid for distances > > half_max_aperture^2/lambda = %f m (propagating at %4.1f)"%
                    (far_field_distance,propagation_distance))
    #
    #compute Fourier transform
    #
    F1 = numpy.fft.fft2(wavefront.get_complex_amplitude())  # Take the fourier transform of the image.
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = numpy.fft.fftshift( F1 )

    # frequency for axis 1
    shape = wavefront.size()
    delta = wavefront.delta()

    pixelsize = delta[0] # p_x[1] - p_x[0]
    npixels = shape[0]
    freq_nyquist = 0.5/pixelsize
    freq_n = numpy.linspace(-1.0,1.0,npixels)
    freq_x = freq_n * freq_nyquist
    freq_x *= wavelength

    print("X: pixelsize %g; npixels=%d, Nyq=%f, fx[]=%g"%(pixelsize,npixels,freq_nyquist,freq_x[0]))

    # frequency for axis 2
    pixelsize = delta[1]
    npixels = shape[1]
    freq_nyquist = 0.5/pixelsize
    freq_n = numpy.linspace(-1.0,1.0,npixels)
    freq_y = freq_n * freq_nyquist
    freq_y *= wavelength

    print("Y: pixelsize %g; npixels=%d, Nyq=%f, fy[]=%g"%(pixelsize,npixels,freq_nyquist,freq_y[0]))
    if propagation_distance != 1.0:
        freq_x *= propagation_distance
        freq_y *= propagation_distance

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(F2,freq_x,freq_y,wavelength=wavelength)
    return  wf_propagated

def propagate_1D_fresnel(wavefront, propagation_distance):
    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :return: a new 2D wavefront object with propagated wavefront
    """

    pass

def propagate_1D_fresnel_convolution(wavefront, propagation_distance):
    """
    2D Fresnel propagator using direct convolution
    :param wavefront:
    :param propagation_distance:
    :return:
    """
    pass

def propagate_2D_integral(wavefront, propagation_distance, detector_abscissas=[None]):
    """
    2D Fresnel-Kirchhoff propagator via simplified integral
    :param wavefront:
    :param propagation_distance: propagation distance
    :param detector_abscissas: a numpy array with the anscissas at the image position. If undefined ([None])
                            it uses the same abscissas present in input wavefront.
    :return: a new 2D wavefront object with propagated wavefront
    """

    pass

#
# tests/example cases
#

def test_propagate_2D_fraunhofer(do_plot=0,wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                                 pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1001,npixels_y=1001,
                                 propagation_distance = 1.0):
    """

    :param do_plot: 0=No plot, 1=Do plot
    :param wavelength:
    :param aperture_type: 'circle' 'square' 'gaussian' (Gaussian sigma = aperture_diameter/2.35)
    :param aperture_diameter:
    :param pixelsize_x:
    :param pixelsize_y:
    :param npixels_x:
    :param npixels_y:
    :param propagation_distance:
    :return:
    """
    print("#                                                            ")
    print("# far field (fraunhofer) diffraction from a square aperture  ")
    print("#                                                            ")

    method = "fraunhofer"
    # method = "fourier_convolution"
    # method = "integral"
    # method = "srw"



    print("Fraunhoffer diffraction valid for distances > > a^2/lambda = %f m"%((aperture_diameter/2)**2/wavelength))

    wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
                                                            x_step=pixelsize_x,
                                                            y_start=-pixelsize_y*npixels_y/2,
                                                            y_step=pixelsize_y,
                                                            wavelength=wavelength,
                                                            number_of_points=(npixels_x,npixels_y))

    wf.set_plane_wave_from_complex_amplitude((1.0+0j))

    if aperture_type == 'circle':
        wf.apply_pinhole(aperture_diameter/2)
    elif aperture_type == 'square':
        wf.apply_slit(-aperture_diameter/2, aperture_diameter/2,-aperture_diameter/2, aperture_diameter/2)
    elif aperture_type == 'gaussian':
        X = wf.get_mesh_x()
        Y = wf.get_mesh_y()
        window = numpy.exp(- (X*X + Y*Y)/2/(aperture_diameter/2.35)**2)
        wf.rescale_amplitudes(window)
    else:
        raise Exception("Not implemented! (accepted: circle, square, gaussian)")



    wf1 = propagate_2D_fraunhofer(wf, propagation_distance)



    if do_plot:
        from srxraylib.plot.gol import plot_image
        plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                   title="aperture intensity (%s), Diameter=%5.1f um"%
                         (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                   show=0)

        plot_image(wf1.get_intensity(),1e6*wf1.get_coordinate_x(),1e6*wf1.get_coordinate_y(),
                   title="Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                         (aperture_type,method,1e6*aperture_diameter),
                   xtitle="X [urad]",ytitle="Y [urad]",
                   show=0)

    # get the theoretical value
    if aperture_type == 'circle': #circular, also display analytical values
        from scipy.special import jv
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        x = (2*numpy.pi/wavelength) * (aperture_diameter/2) * angle_x
        amplitude_theory = 2*jv(1,x)/x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'square':
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        # remove x=0 (to avoid 0/0)
        indices_ok = numpy.where(angle_x != 0)
        angle_x = angle_x[indices_ok]

        x = (2*numpy.pi / wavelength) * (aperture_diameter / 2) * angle_x
        amplitude_theory = 2 * numpy.sin(x)  / x
        intensity_theory = amplitude_theory**2
        intensity_theory /= intensity_theory.max()
    elif aperture_type == 'gaussian':
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        sigma = aperture_diameter/2.35
        sigma_ft = 1.0 / sigma * wavelength / (2.0 * numpy.pi)
        # Factor 2.0 is because we wwant intensity (amplitude**2)
        intensity_theory = numpy.exp( -2.0*(angle_x**2/sigma_ft**2/2) )
    else:
        raise Exception("Undefined aperture type (accepted: circle, square, gaussian)")

    if do_plot:
        from srxraylib.plot.gol import plot
        intensity_calculated =  wf1.get_intensity()[:,wf1.size()[1]/2]
        intensity_calculated /= intensity_calculated.max()
        plot(wf1.get_coordinate_x()*1e6,intensity_calculated,
             angle_x*1e6,intensity_theory,
             legend=["Calculated (FT) H profile","Theoretical"],legend_position=(0.95, 0.95),
             title="Fraunhofer Diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                   (aperture_type,aperture_diameter*1e6,wavelength*1e10),
             xtitle="X (urad)", ytitle="Intensity",xrange=[-80,80])


def test_propagate_2D_fresnel(do_plot=0,wavelength=1.24e-10,aperture_diameter=40e-6,wavefront_length=800e-6,distance=3.6,npoints=1000):
    print("#                                                             ")
    print("# near field (fresnel) diffraction from a square aperture     ")
    print("#                                                             ")

    # wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-wavefront_length/2, x_max=wavefront_length/2)
    # wavefront.set_plane_wave_from_complex_amplitude((2.0+1.0j))
    # wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)
    #
    # wavefront_1 = propagate_1D_fresnel(wavefront, distance)
    #
    # wavefront_2a = propagate_1D_fresnel(wavefront, distance/2)
    # wavefront_2a.apply_ideal_lens(distance/2)
    # wavefront_2b = propagate_1D_fresnel(wavefront_2a, distance/2)
    #
    # if do_plot:
    #     from srxraylib.plot.gol import plot
    #     normalized_intensity = wavefront_1.get_intensity()
    #     normalized_intensity /= normalized_intensity.max()
    #
    #     normalized_intensity2 = wavefront_2b.get_intensity()
    #     normalized_intensity2 /= normalized_intensity2.max()
    #
    #     plot(wavefront_1.get_abscissas()*1e6, normalized_intensity,
    #          wavefront_2b.get_abscissas()*1e6, normalized_intensity2,
    #          legend=["Propagated at %3.1f m"%distance,"Focused %3.1f:%3.1f"%(distance/2,distance/2)],legend_position=(1.0,0.95),
    #          title="Fresnel Diffraction of a square slit",
    #          xtitle="X (um)", ytitle="Intensity",xrange=[-60,60])

def test_propagate_2D_fresnel_convolution(do_plot=0,wavelength=1.24e-10,aperture_diameter=40e-6,wavefront_length=800e-6,distance=3.6,npoints=1000):
    print("#                                                             ")
    print("# near field (fresnel) diffraction from a square aperture     ")
    print("#                                                             ")

    # wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-wavefront_length/2, x_max=wavefront_length/2)
    # wavefront.set_plane_wave_from_complex_amplitude((2.0+1.0j))
    # wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)
    #
    # wavefront_1 = propagate_1D_fresnel_convolution(wavefront, distance)
    #
    # wavefront_2a = propagate_1D_fresnel_convolution(wavefront, distance/2)
    # wavefront_2a.apply_ideal_lens(distance/2)
    # wavefront_2b = propagate_1D_fresnel_convolution(wavefront_2a, distance/2)
    #
    # if do_plot:
    #     from srxraylib.plot.gol import plot
    #     normalized_intensity = wavefront_1.get_intensity()
    #     normalized_intensity /= normalized_intensity.max()
    #
    #     normalized_intensity2 = wavefront_2b.get_intensity()
    #     normalized_intensity2 /= normalized_intensity2.max()
    #
    #     plot(wavefront_1.get_abscissas()*1e6, normalized_intensity,
    #          wavefront_2b.get_abscissas()*1e6, normalized_intensity2,
    #          legend=["Propagated at %3.1f m"%distance,"Focused %3.1f:%3.1f"%(distance/2,distance/2)],legend_position=(1.0,0.95),
    #          title="Fresnel Diffraction (VIA CONVOLUTION) of a square slit",
    #          xtitle="X (um)", ytitle="Intensity",xrange=[-60,60])

def test_propagate_2D_integral(do_plot=0,wavelength=1.24e-10,aperture_diameter=40e-6,wavefront_length=800e-6,distance=3.6,npoints=1000):
    print("#                                                             ")
    print("# near field (fresnel-kirchhoff integral) diffraction from a square aperture     ")
    print("#                                                             ")

    # wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-wavefront_length/2, x_max=wavefront_length/2)
    # wavefront.set_plane_wave_from_complex_amplitude((2.0+1.0j))
    # wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)
    #
    # detector_abscissas = numpy.linspace(-60e-6,60e-6,npoints)
    # wavefront_1 = propagate_1D_integral(wavefront, distance, detector_abscissas=detector_abscissas)
    #
    # wavefront_2a = propagate_1D_integral(wavefront, distance/2, detector_abscissas=detector_abscissas)
    # wavefront_2a.apply_ideal_lens(distance/2)
    # wavefront_2b = propagate_1D_integral(wavefront_2a, distance/2, detector_abscissas=detector_abscissas)
    #
    # if do_plot:
    #     from srxraylib.plot.gol import plot
    #
    #     normalized_intensity = wavefront_1.get_intensity()
    #     normalized_intensity /= normalized_intensity.max()
    #
    #     normalized_intensity2 = wavefront_2b.get_intensity()
    #     normalized_intensity2 /= normalized_intensity2.max()
    #
    #     plot(wavefront_1.get_abscissas()*1e6, normalized_intensity,
    #          wavefront_2b.get_abscissas()*1e6, normalized_intensity2,
    #          legend=["Propagated at %3.1f m"%distance,"Focused %3.1f:%3.1f"%(distance/2,distance/2)],legend_position=(1.0,0.95),
    #          title="Fresnel_Kirchhoff integral diffraction of a square slit",
    #          xtitle="X (um)", ytitle="Intensity",xrange=[-60,60])



if __name__ == "__main__":
    do_plot = 1
    test_propagate_2D_fraunhofer(do_plot=do_plot,aperture_type='gaussian')
    # test_propagate_2D_fresnel(do_plot=do_plot)
    # test_propagate_2D_fresnel_convolution(do_plot=do_plot)
    # test_propagate_2D_integral(do_plot=do_plot)
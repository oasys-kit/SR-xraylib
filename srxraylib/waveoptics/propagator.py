import numpy


from srxraylib.util.data_structures import ScaledArray
from srxraylib.waveoptics.wavefront import Wavefront1D

# TODO: check resulting amplitude normalization

def propagate_1D_fraunhofer(wavefront, propagation_distance):
    """
    1D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :return: a new 1D wavefront object with propagated wavefront
    """

    fft = numpy.fft.fft(wavefront.get_complex_amplitude())
    fft2 = numpy.fft.fftshift(fft)

    # frequency for axis 1

    freq_nyquist = 0.5/wavefront.delta()
    # freq_n = numpy.linspace(-1.0,1.0,wavefront.size())
    # freq_x = freq_n * freq_nyquist
    # freq_x *= wavefront.get_wavelength()
    freq_x_delta = 2/(wavefront.size()-1) * freq_nyquist * wavefront.get_wavelength()
    freq_x_offset = -1.0 * freq_nyquist * wavefront.get_wavelength()


    if propagation_distance == 0:
        return Wavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(fft2, freq_x_offset, freq_x_delta))
    else:
        x = freq_x * propagation_distance
        return Wavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(fft2, x[0], x[1]-x[0]))


def propagate_1D_fresnel(wavefront, propagation_distance):
    """
    1D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :return: a new 1D wavefront object with propagated wavefront
    """
    fft_scale = numpy.fft.fftfreq(wavefront.size())/wavefront.delta()

    fft = numpy.fft.fft(wavefront.get_complex_amplitude())
    fft *= numpy.exp((-1.0j) * numpy.pi * wavefront.get_wavelength() * propagation_distance * fft_scale**2)
    ifft = numpy.fft.ifft(fft)

    return Wavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(ifft, wavefront.offset(), wavefront.delta()))


def propagate_1D_fresnel_convolution(wavefront, propagation_distance):
    """
    1D Fresnel propagator using direct convolution
    :param wavefront:
    :param propagation_distance:
    :return:
    """

    # instead of numpy.convolve, this can be used:
    # from scipy.signal import fftconvolve

    kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * wavefront.get_abscissas()**2 / 2 / propagation_distance)
    kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
    kernel /=  1j * wavefront.get_wavelength() * propagation_distance
    tmp = numpy.convolve(wavefront.get_complex_amplitude(),kernel,mode='same')

    return Wavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(tmp, wavefront.offset(), wavefront.delta()))


def propagate_1D_integral(wavefront, propagation_distance, detector_abscissas=[None]):
    """
    1D Fresnel-Kirchhoff propagator via simplified integral
    :param wavefront:
    :param propagation_distance: propagation distance
    :param detector_abscissas: a numpy array with the anscissas at the image position. If undefined ([None])
                            it uses the same abscissas present in input wavefront.
    :return: a new 1D wavefront object with propagated wavefront
    """

    if detector_abscissas[0] == None:
        detector_abscissas = wavefront.get_abscissas()

    # calculate via outer product, it spreads over a lot of memory, but it is OK for 1D
    x1 = numpy.outer(wavefront.get_abscissas(),numpy.ones(detector_abscissas.size))
    x2 = numpy.outer(numpy.ones(wavefront.size()),detector_abscissas)
    r = numpy.sqrt( numpy.power(x1-x2,2) + numpy.power(propagation_distance,2) )
    wavenumber = numpy.pi*2/wavefront.get_wavelength()
    distances_matrix  = numpy.exp(1.j * wavenumber *  r)


    fieldComplexAmplitude = numpy.dot(wavefront.get_complex_amplitude(),distances_matrix)

    return Wavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(fieldComplexAmplitude, \
                            detector_abscissas[0], detector_abscissas[1]-detector_abscissas[0] ))


#
# tests/example cases
#

def test_propagate_1D_fraunhofer(do_plot=0,wavelength=1.24e-10,aperture_diameter=40e-6,wavefront_length=800e-6,distance=0,npoints=1000):
    print("#                                                            ")
    print("# far field (fraunhofer) diffraction from a square aperture  ")
    print("#                                                            ")

    wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-wavefront_length/2, x_max=wavefront_length/2)
    wavefront.set_plane_wave_constant_complex_amplitude((2.0+1.0j))

    wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)

    wavefront_1 = propagate_1D_fraunhofer(wavefront, distance)

    # get the theoretical value
    angle_x = wavefront_1.get_abscissas()
    x = (2*numpy.pi/wavelength) * (aperture_diameter/2) * angle_x
    U_vs_theta_x = 2*numpy.sin(x)/x
    intensity_theory = U_vs_theta_x**2
    intensity_theory /= intensity_theory.max()


    if do_plot == 1:
        import matplotlib.pylab as plt
        f1 = plt.figure(1)
        intensity_calculated =  wavefront_1.get_intensity()
        intensity_calculated /= intensity_calculated.max()
        plt.plot(wavefront_1.get_abscissas()*1e6,intensity_calculated, label="Calculated (FT)")
        plt.plot(wavefront_1.get_abscissas()*1e6,intensity_theory, label="Theoretical")
        plt.title("Fraunhofer Diffraction of a square slit of %3.1f um at wavelength of %3.1f A"%(aperture_diameter*1e6,wavelength*1e10))
        plt.xlabel("X (urad)")
        plt.ylabel("Intensity")
        plt.xlim([-60, 60])
        ax = plt.subplot(111)
        ax.legend(bbox_to_anchor=(0.95, 0.95))
        plt.show()
    elif do_plot == 2:
        from srxraylib.plot.gol import plot
        intensity_calculated =  wavefront_1.get_intensity()
        intensity_calculated /= intensity_calculated.max()
        plot(wavefront_1.get_abscissas()*1e6,intensity_calculated,
             wavefront_1.get_abscissas()*1e6,intensity_theory,
             legend=["Calculated (FT)","Theoretical"],legend_position=(0.95, 0.95),
             title="Fraunhofer Diffraction of a square slit of %3.1f um at wavelength of %3.1f A"%(aperture_diameter*1e6,wavelength*1e10),
             xtitle="X (urad)", ytitle="Intensity",xrange=[-60,60])



def test_propagate_1D_fresnel(do_plot=0,wavelength=1.24e-10,aperture_diameter=40e-6,wavefront_length=800e-6,distance=3.6,npoints=1000):
    print("#                                                             ")
    print("# near field (fresnel) diffraction from a square aperture     ")
    print("#                                                             ")

    wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-wavefront_length/2, x_max=wavefront_length/2)
    wavefront.set_plane_wave_constant_complex_amplitude((2.0+1.0j))
    wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)

    wavefront_1 = propagate_1D_fresnel(wavefront, distance)

    wavefront_2a = propagate_1D_fresnel(wavefront, distance/2)
    wavefront_2a.apply_ideal_lens(distance/2)
    wavefront_2b = propagate_1D_fresnel(wavefront_2a, distance/2)

    if do_plot == 1:
        import matplotlib.pylab as plt
        f1 = plt.figure(1)

        normalized_intensity = wavefront_1.get_intensity()
        normalized_intensity /= normalized_intensity.max()
        plt.plot(wavefront_1.get_abscissas()*1e6, normalized_intensity, label="Propagated at %3.1f m"%distance)

        normalized_intensity2 = wavefront_2b.get_intensity()
        normalized_intensity2 /= normalized_intensity2.max()
        plt.plot(wavefront_2b.get_abscissas()*1e6, normalized_intensity2, label="Focused %3.1f:%3.1f"%(distance/2,distance/2))

        plt.title("Fresnel Diffraction of a square slit")
        plt.xlabel("X (um)")
        plt.ylabel("Intensity")
        plt.xlim([-60, 60])
        ax = plt.subplot(111)
        ax.legend(bbox_to_anchor=(1.0,0.95))
        plt.show()
    elif do_plot == 2:
        from srxraylib.plot.gol import plot
        normalized_intensity = wavefront_1.get_intensity()
        normalized_intensity /= normalized_intensity.max()

        normalized_intensity2 = wavefront_2b.get_intensity()
        normalized_intensity2 /= normalized_intensity2.max()

        plot(wavefront_1.get_abscissas()*1e6, normalized_intensity,
             wavefront_2b.get_abscissas()*1e6, normalized_intensity2,
             legend=["Propagated at %3.1f m"%distance,"Focused %3.1f:%3.1f"%(distance/2,distance/2)],legend_position=(1.0,0.95),
             title="Fresnel Diffraction of a square slit",
             xtitle="X (um)", ytitle="Intensity",xrange=[-60,60])

def test_propagate_1D_fresnel_convolution(do_plot=0,wavelength=1.24e-10,aperture_diameter=40e-6,wavefront_length=800e-6,distance=3.6,npoints=1000):
    print("#                                                             ")
    print("# near field (fresnel) diffraction from a square aperture     ")
    print("#                                                             ")

    wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-wavefront_length/2, x_max=wavefront_length/2)
    wavefront.set_plane_wave_constant_complex_amplitude((2.0+1.0j))
    wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)

    wavefront_1 = propagate_1D_fresnel_convolution(wavefront, distance)

    wavefront_2a = propagate_1D_fresnel_convolution(wavefront, distance/2)
    wavefront_2a.apply_ideal_lens(distance/2)
    wavefront_2b = propagate_1D_fresnel_convolution(wavefront_2a, distance/2)

    if do_plot == 1:
        import matplotlib.pylab as plt
        f1 = plt.figure(1)

        normalized_intensity = wavefront_1.get_intensity()
        normalized_intensity /= normalized_intensity.max()
        plt.plot(wavefront_1.get_abscissas()*1e6, normalized_intensity, label="Propagated at %3.1f m"%distance)

        normalized_intensity2 = wavefront_2b.get_intensity()
        normalized_intensity2 /= normalized_intensity2.max()
        plt.plot(wavefront_2b.get_abscissas()*1e6, normalized_intensity2, label="Focused %3.1f:%3.1f"%(distance/2,distance/2))

        plt.title("Fresnel Diffraction (VIA CONVOLUTION) of a square slit")
        plt.xlabel("X (um)")
        plt.ylabel("Intensity")
        plt.xlim([-60, 60])
        ax = plt.subplot(111)
        ax.legend(bbox_to_anchor=(1.0,0.95))
        plt.show()
    elif do_plot == 2:
        from srxraylib.plot.gol import plot
        normalized_intensity = wavefront_1.get_intensity()
        normalized_intensity /= normalized_intensity.max()

        normalized_intensity2 = wavefront_2b.get_intensity()
        normalized_intensity2 /= normalized_intensity2.max()

        plot(wavefront_1.get_abscissas()*1e6, normalized_intensity,
             wavefront_2b.get_abscissas()*1e6, normalized_intensity2,
             legend=["Propagated at %3.1f m"%distance,"Focused %3.1f:%3.1f"%(distance/2,distance/2)],legend_position=(1.0,0.95),
             title="Fresnel Diffraction (VIA CONVOLUTION) of a square slit",
             xtitle="X (um)", ytitle="Intensity",xrange=[-60,60])

def test_propagate_1D_integral(do_plot=0,wavelength=1.24e-10,aperture_diameter=40e-6,wavefront_length=800e-6,distance=3.6,npoints=1000):
    print("#                                                             ")
    print("# near field (fresnel-kirchhoff integral) diffraction from a square aperture     ")
    print("#                                                             ")

    wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-wavefront_length/2, x_max=wavefront_length/2)
    wavefront.set_plane_wave_constant_complex_amplitude((2.0+1.0j))
    wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)

    detector_abscissas = numpy.linspace(-60e-6,60e-6,npoints)
    wavefront_1 = propagate_1D_integral(wavefront, distance, detector_abscissas=detector_abscissas) #detector_abscissas)

    wavefront_2a = propagate_1D_integral(wavefront, distance/2, detector_abscissas=detector_abscissas)
    wavefront_2a.apply_ideal_lens(distance/2)
    wavefront_2b = propagate_1D_integral(wavefront_2a, distance/2, detector_abscissas=detector_abscissas)

    if do_plot == 1:
        import matplotlib.pylab as plt
        f1 = plt.figure(1)
        normalized_intensity = wavefront_1.get_intensity()
        normalized_intensity /= normalized_intensity.max()


        plt.plot(wavefront_1.get_abscissas()*1e6, normalized_intensity, label="Propagated at %3.1f m"%distance)

        normalized_intensity2 = wavefront_2b.get_intensity()
        normalized_intensity2 /= normalized_intensity2.max()
        plt.plot(wavefront_2b.get_abscissas()*1e6, normalized_intensity2, label="Focused %3.1f:%3.1f"%(distance/2,distance/2))

        plt.title("Fresnel-Kirchhoff integral diffraction of a square slit")
        plt.xlabel("X (um)")
        plt.ylabel("Intensity")
        plt.xlim([-60, 60])
        ax = plt.subplot(111)
        ax.legend(bbox_to_anchor=(1.0,0.95))
        plt.show()
    elif do_plot == 2:
        from srxraylib.plot.gol import plot

        normalized_intensity = wavefront_1.get_intensity()
        normalized_intensity /= normalized_intensity.max()

        normalized_intensity2 = wavefront_2b.get_intensity()
        normalized_intensity2 /= normalized_intensity2.max()

        plot(wavefront_1.get_abscissas()*1e6, normalized_intensity,
             wavefront_2b.get_abscissas()*1e6, normalized_intensity2,
             legend=["Propagated at %3.1f m"%distance,"Focused %3.1f:%3.1f"%(distance/2,distance/2)],legend_position=(1.0,0.95),
             title="Fresnel_Kirchhoff integral diffraction of a square slit",
             xtitle="X (um)", ytitle="Intensity",xrange=[-60,60])



if __name__ == "__main__":

    test_propagate_1D_fraunhofer(do_plot=2)
    test_propagate_1D_fresnel(do_plot=2)
    test_propagate_1D_fresnel_convolution(do_plot=2)
    test_propagate_1D_integral(do_plot=2)
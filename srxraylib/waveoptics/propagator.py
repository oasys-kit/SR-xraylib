import numpy


from srxraylib.util.data_structures import ScaledArray
from srxraylib.waveoptics.wavefront import Wavefront1D

# TODO: check resulting amplitude normalization

def propagate_1D_fraunhofer(wavefront, propagation_distance=0.0, shift_half_pixel=1):
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
    freq_n = numpy.linspace(-1.0,1.0,wavefront.size())
    freq_x = freq_n * freq_nyquist
    freq_x *= wavefront.get_wavelength()


    if shift_half_pixel:
        freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])

    if propagation_distance == 0:
        wf = Wavefront1D.initialize_wavefront_from_arrays(freq_x,fft2,wavelength=wavefront.get_wavelength())
        return wf
    else:
        wf = Wavefront1D.initialize_wavefront_from_arrays(freq_x*propagation_distance,fft2,wavelength=wavefront.get_wavelength())
        return wf


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


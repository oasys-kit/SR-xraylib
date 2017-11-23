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


def propagate_1D_integral(wavefront, propagation_distance, detector_abscissas=[None],
                          method=0,magnification=1.0,npoints_exit=None):
    """
    1D Fresnel-Kirchhoff propagator via integral implemented as sum
    :param wavefront:
    :param propagation_distance: propagation distance
    :param detector_abscissas: a numpy array with the abscissas at the image position. If undefined ([None])
                            it uses the same abscissas present in input wavefront.
    :param method: 0 (default_ makes a loop over detector coordinates, 1: makes matrices (outer products) so
                            it is more memory hungry.
    :param magnification: if detector_abscissas is [None], the detector abscissas range is the input
                            wavefront range times this magnification factor. Default =1
    :param npoints_exit: if detector_abscissas is [None], the number of points of detector abscissas.
                            Default=None meaning that the same number of points than wavefront are used.
    :return: a new 1D wavefront object with propagated wavefront
    """


    if detector_abscissas[0] == None:

        x = wavefront.get_abscissas()
        if npoints_exit is None:
            npoints_exit = x.size
        detector_abscissas = numpy.linspace(magnification*x[0],magnification*x[-1],npoints_exit)


    wavenumber = numpy.pi*2/wavefront.get_wavelength()
    if method == 0:
        x1 = wavefront.get_abscissas()
        x2 = detector_abscissas
        fieldComplexAmplitude = numpy.zeros_like(x2,dtype=complex)
        for ix,x in enumerate(x2):
            r = numpy.sqrt( numpy.power(x1-x,2) + numpy.power(propagation_distance,2) )
            distances_array  = numpy.exp(1.j * wavenumber *  r)
            fieldComplexAmplitude[ix] = (wavefront.get_complex_amplitude() * distances_array).sum()
    elif method==1:
        # calculate via outer product, it spreads over a lot of memory, but it is OK for 1D
        x1 = numpy.outer(wavefront.get_abscissas(),numpy.ones(detector_abscissas.size))
        x2 = numpy.outer(numpy.ones(wavefront.size()),detector_abscissas)
        r = numpy.sqrt( numpy.power(x1-x2,2) + numpy.power(propagation_distance,2) )
        distances_matrix  = numpy.exp(1.j * wavenumber *  r)
        fieldComplexAmplitude = numpy.dot(wavefront.get_complex_amplitude(),distances_matrix)

    return Wavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(fieldComplexAmplitude, \
                            detector_abscissas[0], detector_abscissas[1]-detector_abscissas[0] ))


def propagator1d_fourier_rescaling(wavefront, propagation_distance, m=1):

    shape = wavefront.size()
    delta = wavefront.delta()
    wavenumber = wavefront.get_wavenumber()
    wavelength = wavefront.get_wavelength()

    fft_scale = numpy.fft.fftfreq(shape)/delta

    x = wavefront.get_abscissas()

    x_rescaling = wavefront.get_abscissas() * m

    r1sq = x ** 2 * (1 - m)
    r2sq = x_rescaling ** 2 * (m - 1 / m)
    fsq = (fft_scale ** 2 / m)
    
    Q1 = wavenumber / 2 / propagation_distance * r1sq
    Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance * fsq)
    Q3 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * r2sq)
    
    wavefront.add_phase_shift(Q1)
    
    fft = numpy.fft.fft(wavefront.get_complex_amplitude())
    ifft = numpy.fft.ifft(fft * Q2) * Q3 / numpy.sqrt(m)

    
    return Wavefront1D(wavefront.get_wavelength(),
                       ScaledArray.initialize_from_steps(ifft, m*wavefront.offset(), m*wavefront.delta()))
    

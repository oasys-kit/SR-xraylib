import numpy

#
# implements 2D propagators:
#
#   propagate_2D_fraunhofer: Far field Fraunhofer propagator. TODO: Check phases, not to be used for downstream propagation
#   propagate_2D_integral: Simplification of the Kirchhoff-Fresnel integral. TODO: Very slow and give some problems
#
#
#   propagate_2D_fresnel               \
#   propagate_2D_fresnel_convolution   | Near field Fresnel propagators via convolution in Fourier space. Three methods
#   propagate_2D_fresnel_srw           /
#
#          three methods available: 'fft': fft -> multiply by kernel in freq -> ifft
#                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
#                                   'srw': use the SRW package
#
#
#
# *********************************** IMPORTANT *******************************************
#                RECOMMENDATIONS:
#
#    The fraunhoffer method cannot be used in a compound system (more than one element) and in connection with lenses
#    The integral propagator is extremely slow for 2D, so by default it only calculates the horizontal and vertical
#        profiles. Therefore, it cannot be used with compound systems.
#
#     >>> Prefer propagate_2D_fresnel <<<
#       Prefer EVEN number of bins.
#       Set shift_half_pixel=1 (now the default)
#    Under these circumstances, the results agree very well with SRW
#
#
#

from srxraylib.util.data_structures import ScaledMatrix
from srxraylib.waveoptics.wavefront2D import Wavefront2D


try:
    import srwlib
    SRWLIB_AVAILABLE = True
except:
    SRWLIB_AVAILABLE = False
    print("SRW is not available")

# TODO: check resulting amplitude normalization (fft and srw likely agree, convolution gives too high amplitudes, so needs normalization)

#TODO: add these elements (like in Timm's application)
#   -slit with absorption
#   -two slits
#   -lens with absorption
#   -gold grid
#   -cubelets
#   -boron fiber with tungsten core
#   -zone plate
#   -siements star


def propagate_2D_fraunhofer(wavefront, propagation_distance=1.0,shift_half_pixel=0): #todo: modificato da giovanni
    """
    2D Fraunhofer propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance. If set to zero, the abscissas
                                 of the returned wavefront are in angle (rad)
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
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


    # frequency for axis 1
    shape = wavefront.size()
    delta = wavefront.delta()
    wavenumber = wavefront.get_wavenumber()

    pixelsize = delta[0] # p_x[1] - p_x[0]
    npixels = shape[0]
    fft_scale = numpy.fft.fftfreq(npixels, d=pixelsize)
    fft_scale = numpy.fft.fftshift(fft_scale)
    x2 = fft_scale * propagation_distance * wavelength

    # frequency for axis 2
    pixelsize = delta[1]
    npixels = shape[1]
    fft_scale = numpy.fft.fftfreq(npixels, d=pixelsize)
    fft_scale = numpy.fft.fftshift(fft_scale)
    y2 = fft_scale * propagation_distance * wavelength

    f_x, f_y = numpy.meshgrid(x2, y2, indexing='ij')
    fsq = numpy.fft.fftshift(f_x ** 2 + f_y ** 2)

    P1 = numpy.exp(1.0j * wavenumber * propagation_distance)
    P2 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * fsq)
    P3 = 1.0j * wavelength * propagation_distance

    F1 = numpy.fft.fft2(wavefront.get_complex_amplitude())  # Take the fourier transform of the image.
    #  Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F1 *= P1
    F1 *= P2
    F1 /= P3
    F2 = numpy.fft.fftshift(F1)

    if shift_half_pixel:
        x2 = x2 - 0.5 * numpy.abs(x2[1] - x2[0])
        y2 = y2 - 0.5 * numpy.abs(y2[1] - y2[0])

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(x2, y2, F2, wavelength=wavelength)
    return  wf_propagated


def propagate_2D_fresnel(wavefront, propagation_distance,shift_half_pixel=1):
    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """
    wavelength = wavefront.get_wavelength()

    #
    # convolving with the Fresnel kernel via FFT multiplication
    #
    fft = numpy.fft.fft2(wavefront.get_complex_amplitude())

    # frequency for axis 1
    shape = wavefront.size()
    delta = wavefront.delta()

    pixelsize = delta[0] # p_x[1] - p_x[0]
    npixels = shape[0]
    freq_nyquist = 0.5/pixelsize
    freq_n = numpy.linspace(-1.0,1.0,npixels)
    freq_x = freq_n * freq_nyquist

    # frequency for axis 2
    pixelsize = delta[1]
    npixels = shape[1]
    freq_nyquist = 0.5/pixelsize
    freq_n = numpy.linspace(-1.0,1.0,npixels)
    freq_y = freq_n * freq_nyquist

    if shift_half_pixel:
        freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])
        freq_y = freq_y - 0.5 * numpy.abs(freq_y[1] - freq_y[0])

    freq_xy = numpy.array(numpy.meshgrid(freq_y,freq_x))
    fft *= numpy.exp((-1.0j) * numpy.pi * wavelength * propagation_distance *
                  numpy.fft.fftshift(freq_xy[0]*freq_xy[0] + freq_xy[1]*freq_xy[1]) )

    ifft = numpy.fft.ifft2(fft)

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x(),
                                                                 wavefront.get_coordinate_y(),
                                                                 ifft,
                                                                 wavelength=wavelength)
    return wf_propagated


def propagate_2D_fresnel_convolution(wavefront, propagation_distance,shift_half_pixel=1):
    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :param shift_half_pixel: set to 1 to shift half pixel (recommended using an even number of pixels) Set as default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    from scipy.signal import fftconvolve

    wavelength = wavefront.get_wavelength()

    X = wavefront.get_mesh_x()
    Y = wavefront.get_mesh_y()

    if shift_half_pixel:
        x = wavefront.get_coordinate_x()
        y = wavefront.get_coordinate_y()
        X += 0.5 * numpy.abs( x[0] - x[1] )
        Y += 0.5 * numpy.abs( y[0] - y[1] )

    kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() *
                       (X**2 + Y**2) / 2 / propagation_distance)
    kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
    kernel /= 1j * wavefront.get_wavelength() * propagation_distance

    tmp = fftconvolve(wavefront.get_complex_amplitude(), kernel, mode='same')

    # tmp = convolve(wavefront.get_complex_amplitude(), kernel, mode='full',method='auto')

    # print(">>>>>>>>>>>> kernel: ",kernel.shape)
    # print(">>>>>>>>>>>> wavefront: ", wavefront.get_complex_amplitude().shape)
    # print(">>>>>>>>>>>> result: ", tmp.shape,numpy.arange(0, tmp.shape[1]).shape)

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x(),
                                                                 wavefront.get_coordinate_y(),
                                                                 tmp,
                                                                 wavelength=wavelength)

    # start = 0.75 * (tmp.shape)[0]
    # start = int(start)
    # end = start + (tmp.shape)[0]
    #
    # tmp2 = (numpy.copy(tmp))[start:end,start:end]
    #
    # print(">>>>>>>>>>> tmp2,start,end",tmp2.shape,start,end)
    #
    #
    # wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(numpy.arange(0,tmp2.shape[0]),
    #                                                              numpy.arange(0, tmp2.shape[1]),
    #                                                              tmp2,
    #                                                              wavelength=wavelength)

    return wf_propagated


def propagator2d_fourier_rescaling(wf,propagation_distance,shift_half_pixel=1,m=1):

    wavefront=wf.duplicate()

    wavenumber = wavefront.get_wavenumber()
    wavelength = wavefront.get_wavelength()

    # frequency for axis 1
    shape = wavefront.size()
    delta = wavefront.delta()

    pixelsize = delta[0]  # p_x[1] - p_x[0]
    npixels = shape[0]
    freq_nyquist = 0.5 / pixelsize
    freq_n = numpy.linspace(-1.0, 1.0, npixels)
    freq_x = freq_n * freq_nyquist

    # frequency for axis 2
    pixelsize = delta[1]
    npixels = shape[1]
    freq_nyquist = 0.5 / pixelsize
    freq_n = numpy.linspace(-1.0, 1.0, npixels)
    freq_y = freq_n * freq_nyquist

    if shift_half_pixel:
        freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])
        freq_y = freq_y - 0.5 * numpy.abs(freq_y[1] - freq_y[0])

    f_x, f_y = numpy.meshgrid(freq_x, freq_y, indexing='ij')
    fsq = numpy.fft.fftshift(f_x ** 2 + f_y ** 2)


    x = wavefront.get_mesh_x()
    y = wavefront.get_mesh_y()

    x_rescaling = wavefront.get_mesh_x() * m
    y_rescaling = wavefront.get_mesh_y() * m

    r1sq = x ** 2 + y**2
    r2sq = x_rescaling ** 2 + y_rescaling**2

    Q1 = wavenumber / 2 * (1 - m) / propagation_distance * r1sq
    Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance / m * fsq)
    Q3 = numpy.exp(1.0j * wavenumber / 2 * (m - 1) / (m * propagation_distance) * r2sq)

    wavefront.add_phase_shift(phase_shift= Q1)

    fft = numpy.fft.fft2(wavefront.get_complex_amplitude())

    ifft = numpy.fft.ifft2(fft * Q2) * Q3 / m

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x() * m,   ####################NON SONO SICURO CHE SIA GIUSTO QUELLO CHE RETURN
                                                                 wavefront.get_coordinate_y() * m,
                                                                 ifft,
                                                                 wavelength=wavelength)

    return wf_propagated


def propagator2d_fourier_rescaling_xy(wf,propagation_distance,shift_half_pixel=1, m_x=1, m_y=1):

    wavefront=wf.duplicate()

    wavenumber = wavefront.get_wavenumber()
    wavelength = wavefront.get_wavelength()

    # frequency for axis 1
    shape = wavefront.size()
    delta = wavefront.delta()

    pixelsize = delta[0]  # p_x[1] - p_x[0]
    npixels = shape[0]
    freq_nyquist = 0.5 / pixelsize
    freq_n = numpy.linspace(-1.0, 1.0, npixels)
    freq_x = freq_n * freq_nyquist

    # frequency for axis 2
    pixelsize = delta[1]
    npixels = shape[1]
    freq_nyquist = 0.5 / pixelsize
    freq_n = numpy.linspace(-1.0, 1.0, npixels)
    freq_y = freq_n * freq_nyquist

    if shift_half_pixel:
        freq_x = freq_x - 0.5 * numpy.abs(freq_x[1] - freq_x[0])
        freq_y = freq_y - 0.5 * numpy.abs(freq_y[1] - freq_y[0])

    f_x, f_y = numpy.meshgrid(freq_x, freq_y, indexing='ij')
    fsq = numpy.fft.fftshift(f_x ** 2 / m_x + f_y ** 2 / m_y)

    x = wavefront.get_mesh_x()
    y = wavefront.get_mesh_y()

    x_rescaling = wavefront.get_mesh_x() * m_x
    y_rescaling = wavefront.get_mesh_y() * m_y

    r1sq = x ** 2 * (1 - m_x) + y ** 2 * (1 - m_y)
    r2sq = x_rescaling ** 2 * (m_x - 1 / m_x) + y_rescaling ** 2 * (m_y - 1 / m_y)

    Q1 = wavenumber / 2 / propagation_distance * r1sq
    Q2 = numpy.exp(-1.0j * numpy.pi * wavelength * propagation_distance * fsq)
    Q3 = numpy.exp(1.0j * wavenumber / 2 / propagation_distance * r2sq)

    wavefront.add_phase_shift(Q1)

    fft = numpy.fft.fft2(wavefront.get_complex_amplitude())

    ifft = numpy.fft.ifft2(fft * Q2) * Q3 / numpy.sqrt(m_x * m_y)

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x() * m_x,   ####################NON SONO SICURO CHE SIA GIUSTO QUELLO CHE RETURN
                                                                 wavefront.get_coordinate_y() * m_y,
                                                                 ifft,
                                                                 wavelength=wavelength)

    return wf_propagated


def propagate_2D_fresnel_srw(wavefront, propagation_distance,
                             srw_autosetting=0):
    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance:
    :param srw_autosetting:set to 1 for automatic SRW redimensionate wavefront
    :return:
    """

    #
    # convolving with the Fresnel kernel via SRW package
    #


    from srxraylib.waveoptics.NumpyToSRW import numpyArrayToSRWArray, SRWWavefrontFromElectricField, SRWEFieldAsNumpy

    import scipy.constants as codata
    angstroms_to_eV = codata.h*codata.c/codata.e*1e10


    srw_wfr = SRWWavefrontFromElectricField(
                    wavefront.get_coordinate_x()[0],wavefront.get_coordinate_x()[-1],wavefront.get_complex_amplitude(),
                    wavefront.get_coordinate_y()[0],wavefront.get_coordinate_y()[-1],numpy.zeros_like(wavefront.get_complex_amplitude()),
                    angstroms_to_eV/(wavefront.get_wavelength()*1e10), 1.0, 1.0, 1e-3, 1.0, 1e-3)
    #
    # propagation
    #
    optDrift = srwlib.SRWLOptD(propagation_distance) #Drift space


    #Wavefront Propagation Parameters:
    #[0]: Auto-Resize (1) or not (0) Before propagation
    #[1]: Auto-Resize (1) or not (0) After propagation
    #[2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
    #[3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
    #[4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
    #[5]: Horizontal Range modification factor at Resizing (1. means no modification)
    #[6]: Horizontal Resolution modification factor at Resizing
    #[7]: Vertical Range modification factor at Resizing
    #[8]: Vertical Resolution modification factor at Resizing
    #[9]: Type of wavefront Shift before Resizing (not yet implemented)
    #[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
    #[11]: New Vertical wavefront Center position after Shift (not yet implemented)

    if srw_autosetting:
        #                 0  1  2   3  4  5   6   7   8   9 10 11
        propagParDrift = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
    else:
        #                 0  1  2   3  4  5   6   7   8   9 10 11
        propagParDrift = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

    optBL = srwlib.SRWLOptC([optDrift], [propagParDrift]) #"Beamline" - Container of Optical Elements (together with the corresponding wavefront propagation instructions)

    print('   Simulating Electric Field Wavefront Propagation by SRW ... ', end='\n')
    srwlib.srwl.PropagElecField(srw_wfr, optBL)



    wavefront2 = Wavefront2D.initialize_wavefront_from_range(srw_wfr.mesh.xStart, srw_wfr.mesh.xFin,
                                                             srw_wfr.mesh.yStart, srw_wfr.mesh.yFin,
                                                             number_of_points=(srw_wfr.mesh.nx,srw_wfr.mesh.ny),
                                                             wavelength=wavefront.get_wavelength())
    amplitude2 = SRWEFieldAsNumpy(srw_wfr)
    amplitude2 = amplitude2[0,:,:,0]
    wavefront2.set_complex_amplitude(amplitude2)

    return wavefront2

def propagate_2D_integral(wavefront, propagation_distance,
                          shuffle_interval=0,calculate_grid_only=1):

    """
    2D Fresnel-Kirchhoff propagator via simplified integral

    NOTE: this propagator is experimental and much less performant than the ones using Fourier Optics
          Therefore, it is not recommended to use.

    :param wavefront:
    :param propagation_distance: propagation distance
    :param shuffle_interval: it is known that this method replicates the central diffraction spot
                            The distace of the replica is proportional to 1/pixelsize
                            To avoid that, it is possible to change a bit (randomly) the coordinates
                            of the wavefront. shuffle_interval controls this shift: 0=No shift. A typical
                             value can be 1e5.
                             The result shows a diffraction pattern without replica but with much noise.
    :param calculate_grid_only: if set, it calculates only the horizontal and vertical profiles, but returns the
                             full image with the other pixels to zero. This is useful when calculating large arrays,
                             so it is set as the default.
    :return: a new 2D wavefront object with propagated wavefront
    """

    #
    # Fresnel-Kirchhoff integral (neglecting inclination factor)
    #

    if calculate_grid_only == 0:
        #
        # calculation over the whole detector area
        #
        p_x = wavefront.get_coordinate_x()
        p_y = wavefront.get_coordinate_y()
        wavelength = wavefront.get_wavelength()
        amplitude = wavefront.get_complex_amplitude()

        det_x = p_x.copy()
        det_y = p_y.copy()

        p_X = wavefront.get_mesh_x()
        p_Y = wavefront.get_mesh_y()

        det_X = p_X
        det_Y = p_Y


        amplitude_propagated = numpy.zeros_like(amplitude,dtype='complex')

        wavenumber = 2 * numpy.pi / wavelength

        for i in range(det_x.size):
            for j in range(det_y.size):
                if shuffle_interval == 0:
                    rd_x = 0.0
                    rd_y = 0.0
                else:
                    rd_x = (numpy.random.rand(p_x.size,p_y.size)-0.5)*shuffle_interval
                    rd_y = (numpy.random.rand(p_x.size,p_y.size)-0.5)*shuffle_interval

                r = numpy.sqrt( numpy.power(p_X + rd_x - det_X[i,j],2) +
                                numpy.power(p_Y + rd_y - det_Y[i,j],2) +
                                numpy.power(propagation_distance,2) )

                amplitude_propagated[i,j] = (amplitude / r * numpy.exp(1.j * wavenumber *  r)).sum()

        wavefront2 = Wavefront2D.initialize_wavefront_from_arrays(det_x,det_y,amplitude_propagated)

    else:
        x = wavefront.get_coordinate_x()
        y = wavefront.get_coordinate_y()
        X = wavefront.get_mesh_x()
        Y = wavefront.get_mesh_y()
        wavenumber = 2 * numpy.pi / wavefront.get_wavelength()
        amplitude = wavefront.get_complex_amplitude()

        used_indices = wavefront.get_mask_grid(width_in_pixels=(1,1),number_of_lines=(1,1))
        indices_x = wavefront.get_mesh_indices_x()
        indices_y = wavefront.get_mesh_indices_y()

        indices_x_flatten = indices_x[numpy.where(used_indices == 1)].flatten()
        indices_y_flatten = indices_y[numpy.where(used_indices == 1)].flatten()
        X_flatten         =         X[numpy.where(used_indices == 1)].flatten()
        Y_flatten         =         Y[numpy.where(used_indices == 1)].flatten()
        complex_amplitude_propagated = amplitude*0

        print("propagate_2D_integral: Calculating %d points from a total of %d x %d = %d"%(
            X_flatten.size,amplitude.shape[0],amplitude.shape[1],amplitude.shape[0]*amplitude.shape[1]))

        for i in range(X_flatten.size):
            r = numpy.sqrt( numpy.power(wavefront.get_mesh_x() - X_flatten[i],2) +
                            numpy.power(wavefront.get_mesh_y() - Y_flatten[i],2) +
                            numpy.power(propagation_distance,2) )

            complex_amplitude_propagated[indices_x_flatten[i],indices_y_flatten[i]] = (amplitude / r * numpy.exp(1.j * wavenumber *  r)).sum()


        wavefront2 = Wavefront2D.initialize_wavefront_from_arrays(x,y,complex_amplitude_propagated,wavefront.get_wavelength())

    return  wavefront2




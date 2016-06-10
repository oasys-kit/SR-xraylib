import numpy

#
# implements 2D propagators:
#
#   propagate_2D_fraunhofer: Far field Fraunhofer propagator. TODO: Check phases, not to be used for downstream propagation
#   propagate_2D_integral: Simplification of the Kirchhoff-Fresnel integral. TODO: Very slow and and give some problems
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

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(freq_x,freq_y,F2,wavelength=wavelength)
    return  wf_propagated

def propagate_2D_fresnel(wavefront, propagation_distance):
    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
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
    # freq = freq * wavelength
    print("X: pixelsize %g; npixels=%d, Nyq=%f, fx[]=%g"%(pixelsize,npixels,freq_nyquist,freq_x[0]))

    # frequency for axis 2
    pixelsize = delta[1]
    npixels = shape[1]
    freq_nyquist = 0.5/pixelsize
    freq_n = numpy.linspace(-1.0,1.0,npixels)
    freq_y = freq_n * freq_nyquist
    # freq_y = freq_y * wavelength
    print("Y: pixelsize %g; npixels=%d, Nyq=%f, fy[]=%g"%(pixelsize,npixels,freq_nyquist,freq_y[0]))

    freq_xy = numpy.array(numpy.meshgrid(freq_y,freq_x))
    fft *= numpy.exp((-1.0j) * numpy.pi * wavelength * propagation_distance *
                  numpy.fft.fftshift(freq_xy[0]*freq_xy[0] + freq_xy[1]*freq_xy[1]) )

    # freq_xy = numpy.array(numpy.meshgrid(freq_x,freq_y))
    # fft *= numpy.exp((-1.0j) * numpy.pi * wavelength * propagation_distance *
    #               numpy.fft.fftshift(freq_xy[0].T*freq_xy[0].T + freq_xy[1].T*freq_xy[1].T) )

    ifft = numpy.fft.ifft2(fft)

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(ifft,
                                                                 wavefront.get_coordinate_x(),
                                                                 wavefront.get_coordinate_y(),
                                                                 wavelength=wavelength)
    return wf_propagated

def propagate_2D_fresnel_convolution(wavefront, propagation_distance):
    """
    2D Fresnel propagator using convolution via Fourier transform
    :param wavefront:
    :param propagation_distance: propagation distance
    :return: a new 2D wavefront object with propagated wavefront
    """

    from scipy.signal import fftconvolve

    wavelength = wavefront.get_wavelength()

    kernel = numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() *
                       (wavefront.get_mesh_x()**2 + wavefront.get_mesh_y()**2)
                       / 2 / propagation_distance)
    kernel *= numpy.exp(1j*2*numpy.pi/wavefront.get_wavelength() * propagation_distance)
    kernel /=  1j * wavefront.get_wavelength() * propagation_distance
    tmp = fftconvolve(wavefront.get_complex_amplitude(),kernel,mode='same')

    wf_propagated = Wavefront2D.initialize_wavefront_from_arrays(wavefront.get_coordinate_x(),
                                                                 wavefront.get_coordinate_y(),
                                                                 tmp,
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
    try:
        import srwlib
    except:
        raise ImportError("Please install srwlib before attempting to us it")

    from NumpyToSRW import numpyArrayToSRWArray, SRWWavefrontFromElectricField, SRWEFieldAsNumpy


    amplitude = wavefront.get_complex_amplitude()
    p_x = wavefront.get_coordinate_x()
    p_y = wavefront.get_coordinate_y()
    wavelength = wavefront.get_wavelength()


    srw_wfr = SRWWavefrontFromElectricField(p_x[0], p_x[-1], amplitude,
                                  p_y[0], p_y[-1], numpy.zeros_like(amplitude),
                                  12396.0/(wavelength*1e10), 1.0, 1.0, 1e-3, 1.0, 1e-3)

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

    print('   Simulating Electric Field Wavefront Propagation bu SRW ... ', end='\n')
    srwlib.srwl.PropagElecField(srw_wfr, optBL)

    amplitude2 = SRWEFieldAsNumpy(srw_wfr)
    amplitude2 = amplitude2[0,:,:,0]
    print("Amplitude shape before:",amplitude.shape,"; after: ",amplitude2.shape)


    # p_x2 = numpy.linspace(srw_wfr.mesh.xStart, srw_wfr.mesh.xFin, srw_wfr.mesh.nx)
    # p_y2 = numpy.linspace(srw_wfr.mesh.yStart, srw_wfr.mesh.yFin, srw_wfr.mesh.ny)
    # wavefront2 = Wavefront2D.initialize_wavefront_from_arrays(amplitude2,p_x2,p_y2)

    wavefront2 = Wavefront2D.initialize_wavefront_from_range(srw_wfr.mesh.xStart, srw_wfr.mesh.xFin,
                                                             srw_wfr.mesh.yStart, srw_wfr.mesh.yFin,
                                                             number_of_points=(srw_wfr.mesh.nx,srw_wfr.mesh.ny))
    wavefront2.set_complex_amplitude(amplitude2)

    return wavefront2 # p_x2,p_y2,amplitude2

def propagate_2D_integral(wavefront, propagation_distance,
                          shuffle_interval=1e-5):
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
    :return: a new 2D wavefront object with propagated wavefront
    """

    #
    # Fresnel-Kirchhoff integral (neglecting inclination factor)
    #
    p_x = wavefront.get_coordinate_x()
    p_y = wavefront.get_coordinate_y()
    wavelength = wavefront.get_wavelength()
    amplitude = wavefront.get_complex_amplitude()

    det_x = p_x.copy()
    det_y = p_y.copy()

    #
    # manual
    #
    # p_xy = numpy.zeros((2,p_x.size,p_y.size))
    # det_xy = numpy.zeros((2,det_x.size,det_y.size))
    # for i in range(p_x.size):
    #     for j in range(p_y.size):
    #         p_xy[0,i,j] = p_x[i]
    #         p_xy[1,i,j] = p_y[j]
    #         det_xy[0,i,j] = det_x[i]
    #         det_xy[1,i,j] = det_y[j]

    #
    # numpy
    #
    # p_xy = numpy.array(numpy.meshgrid(p_y,p_x))
    # det_xy = numpy.array(numpy.meshgrid(det_y,det_x))

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

    wavefront2 = Wavefront2D.initialize_wavefront_from_arrays(amplitude_propagated,det_x,det_y)
    return wavefront2


#
# tests/example cases
#

def test_propagate_2D_fraunhofer(do_plot=0,wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                                 pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024):
    """

    :param do_plot: 0=No plot, 1=Do plot
    :param wavelength:
    :param aperture_type: 'circle' 'square' 'gaussian' (Gaussian sigma = aperture_diameter/2.35)
    :param aperture_diameter:
    :param pixelsize_x:
    :param pixelsize_y:
    :param npixels_x:
    :param npixels_y:
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



    wf1 = propagate_2D_fraunhofer(wf, propagation_distance=1.0) # propagating at 1 m means the result is like in angles



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

#
# fresnel, via convolution in FT space
#          three methods available: 'fft': fft -> multiply by kernel in freq -> ifft
#                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
#                                   'srw': use the SRW package
def test_propagate_2D_fresnel(do_plot=0,method='fft',
                            wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                            pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                            propagation_distance = 30.0,show=1):


    method_label = "fresnel (%s)"%method
    print("#                                                             ")
    print("# near field fresnel (%s) diffraction from a %s aperture  "%(method_label,aperture_type))
    print("#                                                             ")


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


    if method == 'fft':
        wf1 = propagate_2D_fresnel(wf, propagation_distance)
    elif method == 'convolution':
        wf1 = propagate_2D_fresnel_convolution(wf, propagation_distance)
    elif method == 'srw':
        wf1 = propagate_2D_fresnel_srw(wf, propagation_distance)
    else:
        raise Exception("Not implemented method: %s"%method)


    if do_plot:
        from srxraylib.plot.gol import plot_image
        plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                   title="aperture intensity (%s), Diameter=%5.1f um"%
                         (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                   show=0)

        plot_image(wf1.get_intensity(),
                   1e6*wf1.get_coordinate_x()/propagation_distance,
                   1e6*wf1.get_coordinate_y()/propagation_distance,
                   title="Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                         (aperture_type,method_label,1e6*aperture_diameter),
                   xtitle="X [urad]",ytitle="Y [urad]",
                   show=0)

    # get the theoretical value
    if aperture_type == 'circle': #circular, also display analytical values
        from scipy.special import jv
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        angle_x /= propagation_distance
        x = (2*numpy.pi/wavelength) * (aperture_diameter/2) * angle_x
        amplitude_theory = 2*jv(1,x)/x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'square':
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        angle_x /= propagation_distance
        # remove x=0 (to avoid 0/0)
        indices_ok = numpy.where(angle_x != 0)
        angle_x = angle_x[indices_ok]

        x = (2*numpy.pi / wavelength) * (aperture_diameter / 2) * angle_x
        amplitude_theory = 2 * numpy.sin(x)  / x
        intensity_theory = amplitude_theory**2
        intensity_theory /= intensity_theory.max()
    elif aperture_type == 'gaussian':
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        angle_x /= propagation_distance
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
        plot(wf1.get_coordinate_x()*1e6/propagation_distance,intensity_calculated,
             angle_x*1e6,intensity_theory,
             legend=["%s H profile"%method_label,"Theoretical (far field)"],
             legend_position=(0.95, 0.95),
             title="%s diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                   (method_label,aperture_type,aperture_diameter*1e6,wavelength*1e10),
             xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
             show=show)


def test_propagate_2D_integral(do_plot=0,
                            wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                            pixelsize_x=1e-6*5,pixelsize_y=1e-6*10,npixels_x=1024/5,npixels_y=1024/10,
                            propagation_distance = 30.0,show=1):


    method_label = "Kirchhoff-Fresnel integral"
    print("#                                                             ")
    print("# near field (%s) diffraction from a %s aperture  "%(method_label,aperture_type))
    print("#                                                             ")


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


    wf1 = propagate_2D_integral(wf, propagation_distance)

    if do_plot:
        from srxraylib.plot.gol import plot_image
        plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                   title="aperture intensity (%s), Diameter=%5.1f um"%
                         (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                   show=0)

        plot_image(wf1.get_intensity(),
                   1e6*wf1.get_coordinate_x()/propagation_distance,
                   1e6*wf1.get_coordinate_y()/propagation_distance,
                   title="Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                         (aperture_type,method_label,1e6*aperture_diameter),
                   xtitle="X [urad]",ytitle="Y [urad]",
                   show=0)

    # get the theoretical value
    if aperture_type == 'circle': #circular, also display analytical values
        from scipy.special import jv
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        angle_x /= propagation_distance
        x = (2*numpy.pi/wavelength) * (aperture_diameter/2) * angle_x
        amplitude_theory = 2*jv(1,x)/x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'square':
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        angle_x /= propagation_distance
        # remove x=0 (to avoid 0/0)
        indices_ok = numpy.where(angle_x != 0)
        angle_x = angle_x[indices_ok]

        x = (2*numpy.pi / wavelength) * (aperture_diameter / 2) * angle_x
        amplitude_theory = 2 * numpy.sin(x)  / x
        intensity_theory = amplitude_theory**2
        intensity_theory /= intensity_theory.max()
    elif aperture_type == 'gaussian':
        angle_x = wf1.get_coordinate_x() + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        angle_x /= propagation_distance
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
        plot(wf1.get_coordinate_x()*1e6/propagation_distance,intensity_calculated,
             angle_x*1e6,intensity_theory,
             legend=["%s H profile"%method_label,"Theoretical (far field)"],
             legend_position=(0.95, 0.95),
             title="%s diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                   (method_label,aperture_type,aperture_diameter*1e6,wavelength*1e10),
             xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
             show=show)

def test_lens(do_plot=0,method='fft',
                            wavelength=1.24e-10,
                            pixelsize_x=1e-6,npixels_x=2000,pixelsize_y=1e-6,npixels_y=2000,
                            propagation_distance = 30.0,show=1):


    method_label = "fresnel (%s)"%method
    print("#                                                             ")
    print("# near field fresnel (%s) diffraction and focusing  "%(method_label))
    print("#                                                             ")

    #                               \ |  /
    #   *                           | | |                      *
    #                               / | \
    #   <-------    d  ---------------><---------   d   ------->
    #   d is propagation_distance

    wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
                                                            x_step=pixelsize_x,
                                                            y_start=-pixelsize_y*npixels_y/2,
                                                            y_step=pixelsize_y,
                                                            wavelength=wavelength,
                                                            number_of_points=(npixels_x,npixels_y))

    wf.set_plane_wave_from_complex_amplitude(1.0+0j)

    # set spherical wave at the lens entrance (radius=distance)
    # wf.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)

    # apply lens that will focus at propagation_distance downstream the lens.
    # Note that the vertical is a bit defocused
    focal_length = propagation_distance # / 2
    wf.apply_ideal_lens(focal_length,focal_length)

    # if do_plot:
    #     from srxraylib.plot.gol import plot,plot_image
    #     plot_image(wf.get_phase(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='INCOMING phase (%s)'%method,show=0)
    #     plot_image(wf.get_intensity(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='INCOMING intensity (%s)'%method,show=0)


    # propagation downstream the lens to image plane
    if method == 'fft':
        wf2 = propagate_2D_fresnel(wf, propagation_distance)
    elif method == 'convolution':
        wf2 = propagate_2D_fresnel_convolution(wf, propagation_distance)
    elif method == 'srw':
        wf2 = propagate_2D_fresnel_srw(wf, propagation_distance)
    elif method == 'fraunhofer':
        wf2 = propagate_2D_fraunhofer(wf, propagation_distance)
    else:
        raise Exception("Not implemented method: %s"%method)


    horizontal_profile = wf2.get_intensity()[:,npixels_y/2]
    horizontal_profile /= horizontal_profile.max()
    print("FWHM of the horizontal profile: %g um"%(1e6*line_fwhm(horizontal_profile)*wf2.delta()[0]))
    vertical_profile = wf2.get_intensity()[npixels_x/2,:]
    vertical_profile /= vertical_profile.max()
    print("FWHM of the vertical profile: %g um"%(1e6*line_fwhm(vertical_profile)*wf2.delta()[1]))

    if do_plot:
        from srxraylib.plot.gol import plot,plot_image
        plot_image(wf2.get_intensity(),wf2.get_coordinate_x(),wf2.get_coordinate_y(),title='intensity (%s)'%method,show=0)
        # plot_image(wf2.get_amplitude(),wf2.get_coordinate_x(),wf2.get_coordinate_y(),title='amplitude (%s)'%method,show=0)
        plot_image(wf2.get_phase(),wf2.get_coordinate_x(),wf2.get_coordinate_y(),title='phase (%s)'%method,show=0)

        plot(wf2.get_coordinate_x(),horizontal_profile,
             wf2.get_coordinate_y(),vertical_profile,
             legend=['Horizontal profile','Vertical profile'],title="%s"%method,show=show)

#TODO move this somewhere else
def line_fwhm(line):
    #
    #CALCULATE fwhm in number of abscissas bins (supposed on a regular grid)
    #
    tt = numpy.where(line>=max(line)*0.5)
    if line[tt].size > 1:
        # binSize = x[1]-x[0]
        FWHM = (tt[0][-1]-tt[0][0])
        return FWHM
    else:
        return -1

if __name__ == "__main__":
    do_plot = 1

    # test_propagate_2D_fraunhofer(do_plot=do_plot,aperture_type='gaussian')

    # TODO very slow
    # test_propagate_2D_integral(do_plot=do_plot,aperture_type='circle')

    # test_propagate_2D_fresnel(do_plot=do_plot,method='fft',aperture_type='circle',show=0)
    # test_propagate_2D_fresnel(do_plot=do_plot,method='convolution',aperture_type='circle',show=0)
    # test_propagate_2D_fresnel(do_plot=do_plot,method='srw',aperture_type='circle',show=1)


    # test_lens(method='fraunhofer',do_plot=do_plot,show=0)
    # test_lens(method='fft',do_plot=do_plot,show=0)
    test_lens(method='convolution',do_plot=do_plot,show=1)
    # test_lens(method='srw',do_plot=do_plot,show=1)
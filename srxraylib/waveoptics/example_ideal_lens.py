
import numpy

from srxraylib.waveoptics.wavefront2D import Wavefront2D
from srxraylib.waveoptics.propagator2D import propagate_2D_fraunhofer
from srxraylib.waveoptics.propagator2D import propagate_2D_fresnel, propagate_2D_fresnel_convolution, propagate_2D_fresnel_srw

do_plot = True

if do_plot:
    from srxraylib.plot.gol import plot,plot_image,plot_table

try:
    import srwlib
    SRWLIB_AVAILABLE = True
except:
    SRWLIB_AVAILABLE = False
    print("SRW is not available")


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




def propagation_to_image(wf,do_plot=do_plot,plot_title="Before lens",method='fft',
                            propagation_distance=30.0,defocus_factor=1.0,propagation_steps=1,show=1):


    method_label = "fresnel (%s)"%method
    print("\n#                                                             ")
    print("# near field fresnel (%s) diffraction and focusing  "%(method_label))
    print("#                                                             ")

    #                               \ |  /
    #   *                           | | |                      *
    #                               / | \
    #   <-------    d  ---------------><---------   d   ------->
    #   d is propagation_distance

    print("Incident intensity: ",wf.get_intensity().sum())

    # propagation downstream the lens to image plane
    for i in range(propagation_steps):
        if propagation_steps > 1:
            print(">>> Propagating step %d of %d; propagation_distance=%g m"%(i+1,propagation_steps,
                                                propagation_distance*defocus_factor/propagation_steps))
        if method == 'fft':
            wf = propagate_2D_fresnel(wf, propagation_distance*defocus_factor/propagation_steps)
        elif method == 'convolution':
            wf = propagate_2D_fresnel_convolution(wf, propagation_distance*defocus_factor/propagation_steps)
        elif method == 'srw':
            wf = propagate_2D_fresnel_srw(wf, propagation_distance*defocus_factor/propagation_steps)
        elif method == 'fraunhofer':
            wf = propagate_2D_fraunhofer(wf, propagation_distance*defocus_factor/propagation_steps)
        else:
            raise Exception("Not implemented method: %s"%method)




    horizontal_profile = wf.get_intensity()[:,wf.size()[1]/2]
    horizontal_profile /= horizontal_profile.max()
    print("FWHM of the horizontal profile: %g um"%(1e6*line_fwhm(horizontal_profile)*wf.delta()[0]))
    vertical_profile = wf.get_intensity()[wf.size()[0]/2,:]
    vertical_profile /= vertical_profile.max()
    print("FWHM of the vertical profile: %g um"%(1e6*line_fwhm(vertical_profile)*wf.delta()[1]))

    if do_plot:
        from srxraylib.plot.gol import plot,plot_image
        plot_image(wf.get_intensity(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='intensity (%s)'%method,show=0)
        # plot_image(wf.get_amplitude(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='amplitude (%s)'%method,show=0)
        plot_image(wf.get_phase(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='phase (%s)'%method,show=0)

        plot(wf.get_coordinate_x(),horizontal_profile,
             wf.get_coordinate_y(),vertical_profile,
             legend=['Horizontal profile','Vertical profile'],title="%s %s"%(plot_title,method),show=show)

    print("Output intensity: ",wf.get_intensity().sum())
    return wf,wf.get_coordinate_x(),horizontal_profile



def example_ideal_lens(mode_wavefront_before_lens):

    lens_diameter = 0.002
    npixels_x = 2048
    pixelsize_x = lens_diameter / npixels_x
    print("pixelsize: ",pixelsize_x)


    pixelsize_y = pixelsize_x
    npixels_y = npixels_x

    wavelength = 1.24e-10
    propagation_distance = 30.0
    defocus_factor = 1.0 # 1.0 is at focus
    propagation_steps = 1

    # for Gaussian source
    sigma_x = lens_diameter / 200 # 5e-6
    sigma_y = lens_diameter / 400 # 5e-6

    #
    # initialize wavefronts of dimension equal to the lens
    #
    wf_fft = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                     y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                     number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

    wf_convolution = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                     y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                     number_of_points=(npixels_x,npixels_y),wavelength=wavelength)
    if SRWLIB_AVAILABLE:
        wf_srw = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                     y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                     number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

    #
    # calculate/define wavefront at zero distance downstream the lens
    #
    if mode_wavefront_before_lens == 'convergent spherical':
        # no need to propagate nor define lens
        wf_fft.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)
        wf_convolution.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)
        if SRWLIB_AVAILABLE: wf_srw.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)

    elif mode_wavefront_before_lens == 'divergent spherical with lens':
        # define wavefront at zero distance upstream the lens and apply lens
        focal_length = propagation_distance / 2.

        wf_fft.set_spherical_wave(complex_amplitude=1.0,radius=propagation_distance)
        wf_fft.apply_ideal_lens(focal_length,focal_length)

        wf_convolution.set_spherical_wave(complex_amplitude=1.0,radius=propagation_distance)
        wf_convolution.apply_ideal_lens(focal_length,focal_length)

        if SRWLIB_AVAILABLE:
            wf_srw.set_spherical_wave(complex_amplitude=1.0,radius=propagation_distance)
            wf_srw.apply_ideal_lens(focal_length,focal_length)

    elif mode_wavefront_before_lens == 'plane with lens':
        # define wavefront at zero distance upstream the lens and apply lens
        focal_length = propagation_distance

        wf_fft.set_plane_wave_from_complex_amplitude(1.0+0j)
        wf_fft.apply_ideal_lens(focal_length,focal_length)

        wf_convolution.set_plane_wave_from_complex_amplitude(1.0+0j)
        wf_convolution.apply_ideal_lens(focal_length,focal_length)

        if SRWLIB_AVAILABLE:
            wf_srw.set_plane_wave_from_complex_amplitude(1.0+0j)
            wf_srw.apply_ideal_lens(focal_length,focal_length)

    elif mode_wavefront_before_lens == 'Gaussian with lens':
        # define wavefront at source point, propagate to the lens and apply lens
        X = wf_fft.get_mesh_x()
        Y = wf_fft.get_mesh_y()

        intensity = numpy.exp( - X**2/(2*sigma_x**2)) * numpy.exp( - Y**2/(2*sigma_y**2))


        wf_fft.set_plane_wave_from_complex_amplitude( numpy.sqrt(intensity) )
        wf_convolution.set_plane_wave_from_complex_amplitude( numpy.sqrt(intensity) )
        if SRWLIB_AVAILABLE: wf_srw.set_plane_wave_from_complex_amplitude( numpy.sqrt(intensity) )

        # plot

        plot_image(wf_fft.get_intensity(),wf_fft.get_coordinate_x(),wf_fft.get_coordinate_y(),
                   title="G source",show=1)

        wf_fft, tmp1, tmp2 = propagation_to_image(wf_fft,method='fft',propagation_distance=propagation_distance,
                                              do_plot=0,plot_title="Before lens")
        wf_convolution, tmp1, tmp2 = propagation_to_image(wf_convolution,method='convolution',propagation_distance=propagation_distance,
                                              do_plot=0,plot_title="Before lens")
        if SRWLIB_AVAILABLE:
            wf_srw, tmp1, tmp2 = propagation_to_image(wf_srw,method='srw',propagation_distance=propagation_distance,
                                              do_plot=0,plot_title="Before lens")


        plot_image(wf_fft.get_intensity(),wf_fft.get_coordinate_x(),wf_fft.get_coordinate_y(),
                   title="Before lens fft",show=1)

        plot_image(wf_convolution.get_intensity(),wf_fft.get_coordinate_x(),wf_fft.get_coordinate_y(),
                   title="Before lens convolution",show=1)

        focal_length = propagation_distance / 2

        wf_fft.apply_ideal_lens(focal_length,focal_length)
        wf_convolution.apply_ideal_lens(focal_length,focal_length)
        if SRWLIB_AVAILABLE: wf_srw.apply_ideal_lens(focal_length,focal_length)

    else:
        raise Exception("Unknown mode")



    wf_fft, x_fft, y_fft = propagation_to_image(wf_fft,do_plot=0,method='fft',
                            propagation_steps=propagation_steps,
                            propagation_distance = propagation_distance, defocus_factor=defocus_factor)

    wf_convolution, x_convolution, y_convolution = propagation_to_image(wf_convolution,do_plot=0,method='convolution',
                            propagation_steps=propagation_steps,
                            propagation_distance = propagation_distance, defocus_factor=defocus_factor)
    if SRWLIB_AVAILABLE:
        wf_srw, x_srw, y_srw = propagation_to_image(wf_srw,do_plot=0,method='srw',
                                propagation_steps=propagation_steps,
                                propagation_distance = propagation_distance, defocus_factor=defocus_factor)




    if do_plot:
        if SRWLIB_AVAILABLE:
            x = x_fft
            y = numpy.vstack((y_fft,y_srw,y_convolution))

            plot_table(1e6*x,y,legend=["fft","srw","convolution"],ytitle="Intensity",xtitle="x coordinate [um]",
                       title="Comparison 1:1 focusing "+mode_wavefront_before_lens)
        else:
            x = x_fft
            y = numpy.vstack((y_fft,y_convolution))

            plot_table(1e6*x,y,legend=["fft","convolution"],ytitle="Intensity",xtitle="x coordinate [um]",
                       title="Comparison 1:1 focusing "+mode_wavefront_before_lens)



if __name__ == "__main__":

    mode_wavefront_before_lens = 'convergent spherical'
    # mode_wavefront_before_lens = 'divergent spherical with lens'
    # mode_wavefront_before_lens = 'plane with lens'
    mode_wavefront_before_lens = 'Gaussian with lens'
    example_ideal_lens(mode_wavefront_before_lens)
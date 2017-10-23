import unittest
import numpy


from srxraylib.waveoptics.wavefront2D import Wavefront2D
from srxraylib.waveoptics.propagator2D import propagator2d_fourier_rescaling
from srxraylib.waveoptics.propagator2D import propagator2d_fourier_rescaling_xy
from srxraylib.waveoptics.propagator2D import propagate_2D_fresnel

do_plot = True

if do_plot:
    from srxraylib.plot.gol import plot,plot_image,plot_table

#
# some common tools
#
def get_theoretical_diffraction_pattern(angle_x,
                                        aperture_type='square',aperture_diameter=40e-6,
                                        wavelength=1.24e-10,normalization=True):

    # get the theoretical value
    if aperture_type == 'circle': #circular, also display analytical values
        from scipy.special import jv
        x = (2*numpy.pi/wavelength) * (aperture_diameter/2) * angle_x
        amplitude_theory = 2*jv(1,x)/x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'square':
        # remove x=0 (to avoid 0/0) #TODO: check this
        # indices_ok = numpy.where(angle_x != 0)
        # angle_x = angle_x[indices_ok]
        x = (2*numpy.pi / wavelength) * (aperture_diameter / 2) * angle_x
        amplitude_theory = 2 * numpy.sin(x) / x
        intensity_theory = amplitude_theory**2
    elif aperture_type == 'gaussian':
        sigma = aperture_diameter/2.35
        sigma_ft = 1.0 / sigma * wavelength / (2.0 * numpy.pi)
        # Factor 2.0 is because we wwant intensity (amplitude**2)
        intensity_theory = numpy.exp( -2.0*(angle_x**2/sigma_ft**2/2) )
    else:
        raise Exception("Undefined aperture type (accepted: circle, square, gaussian)")

    if normalization:
        intensity_theory /= intensity_theory.max()

    return intensity_theory

def line_image(image,horizontal_or_vertical='H'):
    if horizontal_or_vertical == "H":
        tmp = image[:,int(image.shape[1]/2)]
    else:
        tmp = image[int(image.shape[0]/2),:]
    return tmp

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

def plot_undulator(wavefront, method, show=1):

    plot_image(wavefront.get_intensity(), wavefront.get_coordinate_x(), wavefront.get_coordinate_y(),
               title='Intensity (%s)' % method,
               show=show)
    plot_image(wavefront.get_phase(), wavefront.get_coordinate_x(), wavefront.get_coordinate_y(),
               title='Phase (%s)' % method, show=show)

    horizontal_profile = line_image(wavefront.get_intensity(), horizontal_or_vertical='H')
    vertical_profile = line_image(wavefront.get_intensity(), horizontal_or_vertical='V')
    horizontal_phase_profile_wf = line_image(wavefront.get_phase(), horizontal_or_vertical='H')

    plot(wavefront.get_coordinate_x(), horizontal_profile,
         wavefront.get_coordinate_y(), vertical_profile,
         legend=['Horizontal profile', 'Vertical profile'], title="%s" % method, show=show)

    plot(wavefront.get_coordinate_x(), horizontal_phase_profile_wf,
         legend=['Horizontal phase profile'], title="%s" % method, show=show)

    print("Output intensity: ", wavefront.get_intensity().sum())

class propagator2DTest(unittest.TestCase):
    #
    # TOOLS
    #


    #
    # Common interface for all methods using fresnel, via convolution in FT space
    #          three methods available: 'fft': fft -> multiply by kernel in freq -> ifft
    #                                   'convolution': scipy.signal.fftconvolve(wave,kernel in space)
    #                                   'srw': use the SRW package
    def propagate_2D_fresnel(self,do_plot=do_plot,method='fft',
                                wavelength=1.24e-10,aperture_type='square',aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,
                                propagation_distance = 30.0,m=1, m_x=1, m_y=1, show=1):


        method_label = "fresnel (%s)"%method
        print("\n#                                                             ")
        print("# 2D near field fresnel (%s) diffraction from a %s aperture  "%(method_label,aperture_type))
        print("#                                                             ")


        # wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
        #                                                         x_step=pixelsize_x,
        #                                                         y_start=-pixelsize_y*npixels_y/2,
        #                                                         y_step=pixelsize_y,
        #                                                         wavelength=wavelength,
        #                                                         number_of_points=(npixels_x,npixels_y))

        wf = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
                                                         y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
                                                         number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

        wf.set_plane_wave_from_complex_amplitude((1.0+0j))

        if aperture_type == 'circle':
            wf.apply_pinhole(aperture_diameter/2)
        elif aperture_type == 'square':
            wf.apply_slit(-aperture_diameter/2, aperture_diameter/2,-aperture_diameter/2, aperture_diameter/2)
        else:
            raise Exception("Not implemented! (accepted: circle, square)")


        if method == 'fft':
            wf1 = propagate_2D_fresnel(wf, propagation_distance)
        elif method == 'rescaling':
            wf1 = propagator2d_fourier_rescaling(wf, propagation_distance, m=m)
        elif method == 'rescaling_xy':
            wf1 = propagator2d_fourier_rescaling_xy(wf, propagation_distance, m_x=m_x, m_y=m_y)
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
        angle_x = wf.get_coordinate_x() / propagation_distance

        intensity_theory = get_theoretical_diffraction_pattern(angle_x,aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                            wavelength=wavelength,normalization=True)


        intensity_calculated =  wf1.get_intensity()[:,int(wf1.size()[1]/2)]
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wf1.get_coordinate_x()*1e6/propagation_distance,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["%s H profile"%method_label,"Theoretical (far field)"],
                 legend_position=(0.60, 0.95),
                 title="%s diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (method_label,aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-20,20],
                 show=show)

        if method == 'rescaling_xy' and do_plot:
            angle_y = wf.get_coordinate_y() / propagation_distance
            intensity_calculated_v = wf1.get_intensity()[int(wf1.size()[1] / 2), :]
            intensity_calculated_v /= intensity_calculated_v.max()


            from srxraylib.plot.gol import plot
            plot(wf1.get_coordinate_y() * 1e6 / propagation_distance, intensity_calculated_v,
                 angle_y * 1e6, intensity_theory,
                 legend=["%s V profile" % method_label, "Theoretical (far field)"],
                 legend_position=(0.60, 0.95),
                 title="%s diffraction of a %s slit of %3.1f um at wavelength of %3.1f A" %
                       (method_label, aperture_type, aperture_diameter * 1e6, wavelength * 1e10),
                 xtitle="Y (urad)", ytitle="Intensity", xrange=[-20, 20],
                 show=show)


        return wf1.get_coordinate_x()/propagation_distance,intensity_calculated,angle_x,intensity_theory

    #
    #
    #
    # def propagation_with_lens(self,do_plot=do_plot,method='fft',
    #                             wavelength=1.24e-10,
    #                             pixelsize_x=1e-6,npixels_x=2000,pixelsize_y=1e-6,npixels_y=2000,
    #                             propagation_distance=30.0,defocus_factor=1.0,propagation_steps=1,show=1):
    #
    #
    #     method_label = "fresnel (%s)"%method
    #     print("\n#                                                             ")
    #     print("# near field fresnel (%s) diffraction and focusing  "%(method_label))
    #     print("#                                                             ")
    #
    #     #                              \  |  /
    #     #   *                           | | |                      *
    #     #                              /  |  \
    #     #   <-------    d  ---------------><---------   d   ------->
    #     #   d is propagation_distance
    #
    #     # wf = Wavefront2D.initialize_wavefront_from_steps(x_start=-pixelsize_x*npixels_x/2,
    #     #                                                         x_step=pixelsize_x,
    #     #                                                         y_start=-pixelsize_y*npixels_y/2,
    #     #                                                         y_step=pixelsize_y,
    #     #                                                         wavelength=wavelength,
    #     #                                                         number_of_points=(npixels_x,npixels_y))
    #
    #     wf = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x*npixels_x/2,x_max=pixelsize_x*npixels_x/2,
    #                                                      y_min=-pixelsize_y*npixels_y/2,y_max=pixelsize_y*npixels_y/2,
    #                                                      number_of_points=(npixels_x,npixels_y),wavelength=wavelength)
    #
    #     spherical_or_plane_and_lens = 0
    #     if spherical_or_plane_and_lens == 0:
    #         # set spherical wave at the lens entrance (radius=distance)
    #         wf.set_spherical_wave(complex_amplitude=1.0,radius=-propagation_distance)
    #     else:
    #         # apply lens that will focus at propagation_distance downstream the lens.
    #         # Note that the vertical is a bit defocused
    #         wf.set_plane_wave_from_complex_amplitude(1.0+0j)
    #         focal_length = propagation_distance # / 2
    #         wf.apply_ideal_lens(focal_length,focal_length)
    #
    #     print("Incident intensity: ",wf.get_intensity().sum())
    #
    #     # propagation downstream the lens to image plane
    #     for i in range(propagation_steps):
    #         if propagation_steps > 1:
    #             print(">>> Propagating step %d of %d; propagation_distance=%g m"%(i+1,propagation_steps,
    #                                                 propagation_distance*defocus_factor/propagation_steps))
    #         if method == 'fft':
    #             wf = propagate_2D_fresnel(wf, propagation_distance*defocus_factor/propagation_steps)
    #         else:
    #             raise Exception("Not implemented method: %s"%method)
    #
    #
    #
    #
    #     horizontal_profile = wf.get_intensity()[:,int(wf.size()[1]/2)]
    #     horizontal_profile /= horizontal_profile.max()
    #     print("FWHM of the horizontal profile: %g um"%(1e6*line_fwhm(horizontal_profile)*wf.delta()[0]))
    #     vertical_profile = wf.get_intensity()[int(wf.size()[0]/2),:]
    #     vertical_profile /= vertical_profile.max()
    #     print("FWHM of the vertical profile: %g um"%(1e6*line_fwhm(vertical_profile)*wf.delta()[1]))
    #
    #     if do_plot:
    #         from srxraylib.plot.gol import plot,plot_image
    #         plot_image(wf.get_intensity(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='intensity (%s)'%method,show=0)
    #         # plot_image(wf.get_amplitude(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='amplitude (%s)'%method,show=0)
    #         plot_image(wf.get_phase(),wf.get_coordinate_x(),wf.get_coordinate_y(),title='phase (%s)'%method,show=0)
    #
    #         plot(wf.get_coordinate_x(),horizontal_profile,
    #              wf.get_coordinate_y(),vertical_profile,
    #              legend=['Horizontal profile','Vertical profile'],title="%s"%method,show=show)
    #
    #     print("Output intensity: ",wf.get_intensity().sum())
    #     return wf.get_coordinate_x(),horizontal_profile


    def undulator(self, method, npixels_x, npixels_y,
                  pixelsize_x, pixelsize_y,
                  wavelength, sigma,
                  m, m_x, m_y,
                  propagation_distance, radius, show,
                  image_source, apply_lens):

        wavefront = Wavefront2D.initialize_wavefront_from_range(x_min=-pixelsize_x * npixels_x / 2,
                                                                x_max=pixelsize_x * npixels_x / 2,
                                                                y_min=-pixelsize_y * npixels_y / 2,
                                                                y_max=pixelsize_y * npixels_y / 2,
                                                                number_of_points=(npixels_x, npixels_y),
                                                                wavelength=wavelength)

        wavefront.set_spherical_wave(radius=radius)

        if image_source == False:

            X = wavefront.get_mesh_x()
            Y = wavefront.get_mesh_y()

            wavefront.rescale_amplitude(numpy.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)))  # applichiamo intensita' con profilo gaussiano

        print("Input intensity: ", wavefront.get_intensity().sum())

        if apply_lens == True:
            wavefront.apply_ideal_lens(focal_length_x=propagation_distance, focal_length_y=propagation_distance)

            print("Intensity after lens: ", wavefront.get_intensity().sum())

        if method == 'fft':
            output_wf = propagate_2D_fresnel(wavefront, propagation_distance=propagation_distance)
        elif method == 'rescaling':
            output_wf = propagator2d_fourier_rescaling(wavefront, propagation_distance=propagation_distance, m=m)
        elif method == 'rescaling_xy':
            output_wf = propagator2d_fourier_rescaling_xy(wavefront, propagation_distance=propagation_distance,
                                                                   m_x=m_x, m_y=m_y)

        plot_undulator(output_wf, method=method, show=show)

        horizontal_profile_wf = line_image(wavefront.get_intensity(), horizontal_or_vertical='H')

        vertical_profile_wf = line_image(wavefront.get_intensity(),
                                           horizontal_or_vertical='V')

        horizontal_phase_profile_wf = line_image(wavefront.get_phase(), horizontal_or_vertical='H')

        if image_source == True:
            return wavefront.get_coordinate_x(), horizontal_profile_wf, horizontal_phase_profile_wf


    #
    # TESTS
    #

    def test_propagate_2D_fresnel_square(self):
        xcalc, ycalc, xtheory, ytheory = self.propagate_2D_fresnel(do_plot=True,method='rescaling_xy',aperture_type='square',
                                aperture_diameter=40e-6,
                                pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=2048,npixels_y=2048,
                                propagation_distance=30.0,wavelength=1.24e-10,m=1,m_x=0.5,m_y=1)
        # methods: fft, rescaling, rescaling_xy
        # aperture: square, circle
        numpy.testing.assert_almost_equal(ycalc/10,ytheory/10,1)

    def test_undulator (self):

        dimensione_screen_x = 5e-3
        dimensione_screen_y = 2e-3

        npixels_x = 2048
        npixels_y = 2048

        pixelsize_x = dimensione_screen_x / npixels_x
        pixelsize_y = dimensione_screen_y / npixels_y

        wavelength = 73e-12

        sigma = 140e-6

        m = 1
        m_x = 1
        m_y = 1

        focal_length = 8.27775

        show_graph = True
        # show_graph = True

        radius = 28.3

        method = 'fft'
        # method = 'rescaling'
        # method = 'rescaling_xy'

        ### NON STO CONSIDERANDO CHE LA LUNGHEZZA FOCALE SU Y E' INFINITA, TODO

        self.undulator(method,npixels_x=npixels_x, npixels_y=npixels_y,
                       pixelsize_x=pixelsize_x, pixelsize_y=pixelsize_y,
                       wavelength=wavelength, sigma=sigma,
                       m=m, m_x=m_x, m_y=m_y,
                       propagation_distance=focal_length, radius=radius, show=show_graph,
                       image_source=False, apply_lens=True)

    def test_comparison_source_vs_propagation (self):

        dimensione_screen_x = 5e-3
        dimensione_screen_y = dimensione_screen_x

        npixels_x = 2048
        npixels_y = npixels_x

        pixelsize_x = dimensione_screen_x / npixels_x
        pixelsize_y = pixelsize_x

        wavelength = 73e-12

        sigma = 140e-6

        m = 1
        m_x = 1
        m_y = 1

        propagation_distance = 30

        # show_graph = False
        show_graph = True

        radius = 30

        # method = 'fft'
        method = 'rescaling'
        # method = 'rescaling_xy'

        cord_x_forward, intensity_x_forward, phase_x_forward = self.undulator(method, npixels_x=npixels_x, npixels_y=npixels_y,
                       pixelsize_x=pixelsize_x, pixelsize_y=pixelsize_y,
                       wavelength=wavelength, sigma=sigma,
                       m=m, m_x=m_x, m_y=m_y,
                       propagation_distance=propagation_distance/2, radius=radius, show=show_graph,
                       image_source=True, apply_lens= True)

        cord_x_backward, intensity_x_backward, phase_x_backward = self.undulator(method, npixels_x=npixels_x, npixels_y=npixels_y,
                       pixelsize_x=pixelsize_x, pixelsize_y=pixelsize_y,
                       wavelength=wavelength, sigma=sigma,
                       m=m, m_x=m_x, m_y=m_y,
                       propagation_distance=propagation_distance, radius=-radius, show=show_graph,
                       image_source=True, apply_lens = False)

        numpy.testing.assert_almost_equal(intensity_x_forward, intensity_x_backward, 1)
        numpy.testing.assert_almost_equal(phase_x_forward, phase_x_backward, 1)

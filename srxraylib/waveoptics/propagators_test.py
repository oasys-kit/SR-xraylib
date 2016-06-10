import unittest
import numpy

from srxraylib.waveoptics.wavefront import Wavefront1D
from srxraylib.waveoptics.wavefront2D import Wavefront2D

from srxraylib.waveoptics.propagator import propagate_1D_fraunhofer
from srxraylib.waveoptics.propagator2D import propagate_2D_fraunhofer

do_plot = 1

if do_plot:
    from srxraylib.plot.gol import plot,plot_image


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
        indices_ok = numpy.where(angle_x != 0)
        angle_x = angle_x[indices_ok]
        x = (2*numpy.pi / wavelength) * (aperture_diameter / 2) * angle_x
        amplitude_theory = 2 * numpy.sin(x)  / x
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

class propagatorTest(unittest.TestCase):
    #
    # tests/example cases for 1D propagators
    #
    def test_propagate_1D_fraunhofer(self,do_plot=do_plot,aperture_type='square',aperture_diameter=40e-6,
                    wavefront_length=800e-6,npoints=1024,wavelength=1.24e-10,):
        print("#                                                            ")
        print("# far field 1D (fraunhofer) diffraction from a %s aperture  "%aperture_type)
        print("#                                                            ")

        wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints,
                                                                x_min=-wavefront_length/2, x_max=wavefront_length/2)
        wavefront.set_plane_wave_from_complex_amplitude((2.0+1.0j))

        if aperture_type == 'square':
            wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)
        else:
            raise Exception("Undefinded aperture type: %s"%aperture_type)

        wavefront_1 = propagate_1D_fraunhofer(wavefront)


        intensity_theory = get_theoretical_diffraction_pattern(wavefront_1.get_abscissas(),
                                                    aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                                    wavelength=wavelength,normalization=True)

        intensity_calculated =  wavefront_1.get_intensity()
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            plot(wavefront_1.get_abscissas()*1e6,intensity_calculated,
                 wavefront_1.get_abscissas()*1e6,intensity_theory,
                 legend=["Calculated (FT)","Theoretical"],legend_position=(0.95, 0.95),
                 title="1D Fraunhofer Diffraction of a square slit of %3.1f um at wavelength of %3.1f A"%(aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-60,60])

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)


class propagator2DTest(unittest.TestCase):

    def test_propagate_2D_fraunhofer(self,do_plot=do_plot,aperture_type='square',aperture_diameter=40e-6,
                    pixelsize_x=1e-6,pixelsize_y=1e-6,npixels_x=1024,npixels_y=1024,wavelength=1.24e-10):
        """

        :param do_plot: 0=No plot, 1=Do plot
        :param aperture_type: 'circle' 'square' 'gaussian' (Gaussian sigma = aperture_diameter/2.35)
        :param aperture_diameter:
        :param pixelsize_x:
        :param pixelsize_y:
        :param npixels_x:
        :param npixels_y:
        :param wavelength:
        :return:
        """

        print("#                                                            ")
        print("# far field 2D (fraunhofer) diffraction from a square aperture  ")
        print("#                                                            ")

        method = "fraunhofer"

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
            plot_image(wf.get_intensity(),1e6*wf.get_coordinate_x(),1e6*wf.get_coordinate_y(),
                       title="aperture intensity (%s), Diameter=%5.1f um"%
                             (aperture_type,1e6*aperture_diameter),xtitle="X [um]",ytitle="Y [um]",
                       show=0)

            plot_image(wf1.get_intensity(),1e6*wf1.get_coordinate_x(),1e6*wf1.get_coordinate_y(),
                       title="2D Diffracted intensity (%s) by a %s slit of aperture %3.1f um"%
                             (aperture_type,method,1e6*aperture_diameter),
                       xtitle="X [urad]",ytitle="Y [urad]",
                       show=0)

        angle_x = wf1.get_coordinate_x() # + 0.5*wf1.delta()[0] # shifted of half-pixel!!!
        intensity_theory = get_theoretical_diffraction_pattern(angle_x,
                                            aperture_type=aperture_type,aperture_diameter=aperture_diameter,
                                            wavelength=wavelength,normalization=True)


        intensity_calculated =  wf1.get_intensity()[:,wf1.size()[1]/2]
        intensity_calculated /= intensity_calculated.max()

        if do_plot:
            plot(wf1.get_coordinate_x()*1e6,intensity_calculated,
                 angle_x*1e6,intensity_theory,
                 legend=["Calculated (FT) H profile","Theoretical"],legend_position=(0.95, 0.95),
                 title="2D Fraunhofer Diffraction of a %s slit of %3.1f um at wavelength of %3.1f A"%
                       (aperture_type,aperture_diameter*1e6,wavelength*1e10),
                 xtitle="X (urad)", ytitle="Intensity",xrange=[-80,80])

        numpy.testing.assert_almost_equal(intensity_calculated,intensity_theory,1)



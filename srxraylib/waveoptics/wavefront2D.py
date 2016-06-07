
import numpy


from srxraylib.util.data_structures import ScaledMatrix

#------------------------------------------------
#
#
#
#
#------------------------------------------------

class Wavefront2D(object):
    wavelength = 0.0
    electric_field_array = None

    def __init__(self, wavelength=1e-10, electric_field_array=None):
        self.wavelength = wavelength
        self.electric_field_array = electric_field_array

    @classmethod
    def initialize_wavefront(cls, wavelength=1e-10, number_of_points=(100,100)):
        return Wavefront2D(wavelength, ScaledMatrix.initialize(
            np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),interpolator=True))

    @classmethod
    def initialize_wavefront_from_steps(cls, x_start=0.0, x_step=0.0, y_start=0.0, y_step=0.0,
                                        wavelength=1e-10, number_of_points=(100,100),):
        sM = ScaledMatrix.initialize_from_steps(
                    numpy.full(number_of_points,(1.0 + 0.0j), dtype=complex),
                    x_start,x_step,y_start,y_step,interpolator=True)

        return Wavefront2D(wavelength,sM)

    @classmethod
    def initialize_wavefront_from_range(cls, x_min=0.0, x_max=0.0, y_min=0.0, y_max=0.0,
                                        wavelength=1e-10, number_of_points=(100,100), ):
        return Wavefront2D(wavelength, ScaledArray.initialize_from_range( \
                    numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                    x_min,x_max,y_min,y_max),interpolator=True)


    # main parameters

    def size(self):
        return self.electric_field_array.shape()

    def delta(self):
        x = self.get_coordinate_x()
        y = self.get_coordinate_y()
        return x[1]-x[0],y[1]-y[0]
    #
    def offset(self):
        return self.get_coordinate_x()[0],self.get_coordinate_y()[0]

    def get_wavelength(self):
        return self.wavelength

    def get_wavenumber(self):
        return 2*numpy.pi/self.wavelength

    def get_coordinate_x(self):
        return self.electric_field_array.get_x_values()

    def get_coordinate_y(self):
        return self.electric_field_array.get_y_values()


    def get_complex_amplitude(self):
        return self.electric_field_array.get_z_values()

    def get_amplitude(self):
        return numpy.absolute(self.get_complex_amplitude())

    def get_phase(self):
        return numpy.arctan2(numpy.imag(self.get_complex_amplitude()), numpy.real(self.get_complex_amplitude()))

    def get_intensity(self):
        return self.get_amplitude()**2


    # interpolated values (a bit redundant, but kept the same interfacs as wavefront 1D)

    def get_interpolated(self,x_value,y_value,toreturn='complex_amplitude'):
        if self.electric_field_array.interpolator == False:
            raise Exception("Interpolator not available!")
        interpolated_values = self.electric_field_array.interpolate_value(x_value,y_value)
        if toreturn == 'complex_amplitude':
            return interpolated_values
        elif toreturn == 'amplitude':
            return numpy.abs(interpolated_values)
        elif toreturn == 'phase':
            return numpy.arctan2(numpy.imag(interpolated_values), numpy.real(interpolated_values))
        elif toreturn == 'intensity':
            return numpy.abs(interpolated_values)**2
        else:
            raise Exception('Unknown return string')

    def get_interpolated_complex_amplitude(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='complex_amplitude')

    def get_interpolated_complex_amplitudes(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='complex_amplitude')

    def get_interpolated_amplitude(self, x_value,y_value): # singular!
        return self.get_interpolated(x_value,y_value,toreturn='amplitude')

    def get_interpolated_amplitudes(self, x_value,y_value): # plural!
        return self.get_interpolated(x_value,y_value,toreturn='amplitude')
    #
    def get_interpolated_phase(self, x_value,y_value): # singular!
        return self.get_interpolated(x_value,y_value,toreturn='phase')

    def get_interpolated_phases(self, x_value,y_value): # plural!
        return self.get_interpolated(x_value,y_value,toreturn='phase')

    def get_interpolated_intensity(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='intensity')

    def get_interpolated_intensities(self, x_value,y_value):
        return self.get_interpolated(x_value,y_value,toreturn='intensity')


    # modifiers

    def set_plane_wave_from_complex_amplitude(self, complex_amplitude=(1.0 + 0.0j)):
        new_value = self.electric_field_array.get_z_values()
        new_value *= 0.0
        new_value += complex_amplitude
        self.electric_field_array.set_z_values(new_value)
        self.electric_field_array.compute_interpolator()

    def set_plane_wave_from_amplitude_and_phase(self, amplitude=1.0, phase=0.0):
        self.set_plane_wave_from_complex_amplitude(amplitude*numpy.cos(phase) + 1.0j*amplitude*numpy.sin(phase))
        self.electric_field_array.compute_interpolator()

    def set_spherical_wave(self, amplitude=1.0, radius=1.0):
        if radius == 0:
            raise Exception("Radius cannot be zero")
        X = self.electric_field_array.get_x_values()
        Y = self.electric_field_array.get_y_values()
        XY = numpy.meshgrid(X,Y)
        new_value = (amplitude/radius)*numpy.exp(-1.0j*self.get_wavenumber()*(XY[0]**2+XY[1]**2)/(2*radius))
        self.electric_field_array.set_z_values(new_value)
        self.electric_field_array.compute_interpolator()

    def add_phase_shift(self, phase_shift):
        new_value = self.electric_field_array.get_z_values()
        new_value *= numpy.exp(1.0j*phase_shift)
        self.electric_field_array.set_z_values(new_value)
        self.electric_field_array.compute_interpolator()

    def add_phase_shifts(self, phase_shifts):
        if phase_shifts.shape != self.electric_field_array.shape():
            raise Exception("Phase Shifts array has different dimension")
        new_value = self.electric_field_array.get_z_values()
        new_value *= numpy.exp(1.0j*phase_shifts)
        self.electric_field_array.set_z_values(new_value)
        self.electric_field_array.compute_interpolator()

    def rescale_amplitude(self, factor):
        new_value = self.electric_field_array.get_z_values()
        new_value *= factor
        self.electric_field_array.set_z_values(new_value)
        self.electric_field_array.compute_interpolator()

    def rescale_amplitudes(self, factors):
        if factors.shape != self.electric_field_array.shape():
            raise Exception("Factors array has different dimension")
        new_value = self.electric_field_array.get_z_values()
        new_value *= factors
        self.electric_field_array.set_z_values(new_value)
        self.electric_field_array.compute_interpolator()

    def apply_ideal_lens(self, focal_length_x, focal_length_y):
        X = self.electric_field_array.get_x_values()
        Y = self.electric_field_array.get_y_values()
        XY = numpy.meshgrid(X,Y)
        self.add_phase_shifts((-1.0) * self.get_wavenumber() * ( ( (XY[0]**2)/focal_length_x + (XY[1]**2)/focal_length_y) / 2))

    def apply_slit(self, x_slit_min, x_slit_max, y_slit_min, y_slit_max):
        window = numpy.ones(self.electric_field_array.shape())

        lower_window_x = numpy.where(self.get_coordinate_x() < x_slit_min)
        upper_window_x = numpy.where(self.get_coordinate_x() > x_slit_max)
        lower_window_y = numpy.where(self.get_coordinate_y() < y_slit_min)
        upper_window_y = numpy.where(self.get_coordinate_y() > y_slit_max)

        if len(lower_window_x) > 0: window[lower_window_x,:] = 0
        if len(upper_window_x) > 0: window[upper_window_x,:] = 0
        if len(lower_window_y) > 0: window[:,lower_window_y] = 0
        if len(upper_window_y) > 0: window[:,upper_window_y] = 0

        self.rescale_amplitudes(window)

    # new
    def set_complex_amplitude(self,complex_amplitude):
        if self.electric_field_array.shape() != complex_amplitude.shape:
            raise Exception("Incompatible shape")
        self.electric_field_array.set_z_values(complex_amplitude)
        self.electric_field_array.compute_interpolator()

def test_plane_wave(do_plot=0):
    #
    # plane wave
    #

    wavelength        = 1.24e-10

    wavefront_length_x = 400e-6
    wavefront_length_y = wavefront_length_x

    npixels_x =  1024
    npixels_y =  npixels_x

    pixelsize_x = wavefront_length_x / npixels_x
    pixelsize_y = wavefront_length_y / npixels_y



    wavefront = Wavefront2D.initialize_wavefront_from_steps(
                    x_start=-0.5*wavefront_length_x,x_step=pixelsize_x,
                    y_start=-0.5*wavefront_length_y,y_step=pixelsize_y,
                    number_of_points=(npixels_x,npixels_y),wavelength=wavelength)

    # possible modifications

    wavefront.set_plane_wave_from_amplitude_and_phase(5.0,numpy.pi/2)

    wavefront.add_phase_shift(numpy.pi/2)

    wavefront.rescale_amplitude(10.0)

    wavefront.set_spherical_wave(1,10e-6)

    wavefront.rescale_amplitudes(numpy.ones_like(wavefront.get_intensity())*0.1)

    wavefront.apply_ideal_lens(5.0,10.0)

    wavefront.apply_slit(-50e-6,10e-6,-20e-6,40e-6)

    wavefront.set_plane_wave_from_complex_amplitude(2.0+3j)


    print("Wevefront X value",wavefront.get_coordinate_x())
    print("Wavefront Y value",wavefront.get_coordinate_y())

    print("wavefront intensity",wavefront.get_intensity())
    print("Wavefront complex ampl",wavefront.get_complex_amplitude())
    print("Wavefront phase",wavefront.get_phase())


    if do_plot:
        from srxraylib.plot.gol import plot_image
        plot_image(wavefront.get_intensity(),wavefront.get_coordinate_x(),wavefront.get_coordinate_y(),
                   title="Intensity",show=0)
        plot_image(wavefront.get_phase(),wavefront.get_coordinate_x(),wavefront.get_coordinate_y(),
                   title="Phase",show=1)


def test_interpolator(do_plot=0):
    #
    # interpolator
    #

    x = numpy.linspace(-10,10,100)
    y = numpy.linspace(-20,20,50)
    XY = numpy.meshgrid(y,x)
    sigma = 3.0
    Z = numpy.exp(- (XY[0]**2+XY[1]**2)/2/sigma**2)
    print("???? Z",Z.shape)

    wf = Wavefront2D.initialize_wavefront_from_steps(x[0],x[1]-x[0],y[0],y[1]-y[0],number_of_points=(100,50))
    print("wf shape: ",wf.size())
    wf.set_complex_amplitude( Z )

    x1 = 3.2
    y1 = -2.5
    z1 = numpy.exp(- (x1**2+y1**2)/2/sigma**2)
    print("complex ampl at (%g,%g): %g+%gi (exact=%g)"%(x1,y1,
                                                    wf.get_interpolated_complex_amplitude(x1,y1).real,
                                                    wf.get_interpolated_complex_amplitude(x1,y1).imag,
                                                    z1))
    print("intensity  at (%g,%g):   %g (exact=%g)"%(x1,y1,wf.get_interpolated_intensity(x1,y1),z1**2))




    if do_plot:
        from srxraylib.plot.gol import plot_image
        plot_image(wf.get_intensity(),wf.get_coordinate_x(),wf.get_coordinate_y(),title="Original",show=0)
        plot_image(wf.get_interpolated_intensity(XY[0],XY[1]),wf.get_coordinate_x(),wf.get_coordinate_y(),
                   title="interpolated on same grid",show=1)

if __name__ == "__main__":

    test_plane_wave(do_plot=0)
    test_interpolator(do_plot=1)












import numpy
import warnings

from srxraylib.util.data_structures import ScaledArray
import scipy.constants as codata


# TODO: add polarization (like for 2D)

#------------------------------------------------
#
#
#
#
#------------------------------------------------

class Wavefront1D(object):
    wavelength = 0.0
    electric_field_array = None

    def __init__(self, wavelength=1e-10, electric_field_array=None):
        self.wavelength = wavelength
        self.electric_field_array = electric_field_array

    @classmethod
    def initialize_wavefront(cls, wavelength=1e-10, number_of_points=1000):
        return Wavefront1D(wavelength, ScaledArray.initialize(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex)))

    @classmethod
    def initialize_wavefront_from_steps(cls, x_start=0.0, x_step=0.0, number_of_points=1000, wavelength=1e-10):
        return Wavefront1D(wavelength, ScaledArray.initialize_from_steps(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         initial_scale_value=x_start,
                                                                         scale_step=x_step))
    @classmethod
    def initialize_wavefront_from_range(cls, x_min=0.0, x_max=0.0, number_of_points=1000, wavelength=1e-10 ):
        return Wavefront1D(wavelength, ScaledArray.initialize_from_range(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         min_scale_value=x_min,
                                                                         max_scale_value=x_max))

    @classmethod
    def initialize_wavefront_from_arrays(cls, x_array, y_array, wavelength=1e-10,):
        if x_array.size != y_array.size:
            raise Exception("Unmatched shapes for x and y")

        return Wavefront1D(wavelength, ScaledArray.initialize_from_steps(np_array=y_array,
                                                                         initial_scale_value=x_array[0],
                                                                         scale_step=numpy.abs(x_array[1]-x_array[0])))


    # main parameters

    def duplicate(self):
        return self.initialize_wavefront_from_arrays(
            x_array=self.get_abscissas(), y_array=self.get_complex_amplitude(),
            wavelength=self.get_wavelength()) # todo: aggiunto da giovanni, controllare

    def size(self):
        return self.electric_field_array.size()

    def delta(self):
        return self.electric_field_array.delta()

    def offset(self):
        return self.electric_field_array.offset()

    def get_wavelength(self):
        return self.wavelength

    def get_wavenumber(self):
        return 2*numpy.pi/self.wavelength

    def get_abscissas(self):
        return self.electric_field_array.scale

    def get_complex_amplitude(self):
        return self.electric_field_array.np_array

    def get_amplitude(self):
        return numpy.absolute(self.get_complex_amplitude())

    def get_phase(self,from_minimum_intensity=0.0):
        phase = numpy.angle(self.get_complex_amplitude())
        if (from_minimum_intensity > 0.0):
            intensity = self.get_intensity()
            intensity /= intensity.max()
            bad_indices = numpy.where(intensity < from_minimum_intensity )
            phase[bad_indices] = 0.0

        return phase

    def get_intensity(self):
        return self.get_amplitude()**2

    def get_normalized_intensity(self): # todo: aggiunto da giovanni
        return self.get_intensity() / self.get_intensity().max()

    # interpolated values

    def get_interpolated_complex_amplitude(self, abscissa_value): # singular
        return self.electric_field_array.interpolate_value(abscissa_value)

    def get_interpolated_complex_amplitudes(self, abscissa_values): # plural
        return self.electric_field_array.interpolate_values(abscissa_values)

    def get_interpolated_amplitude(self, abscissa_value): # singular!
        return numpy.absolute(self.get_interpolated_complex_amplitude(abscissa_value))

    def get_interpolated_amplitudes(self, abscissa_values): # plural!
        return numpy.absolute(self.get_interpolated_complex_amplitudes(abscissa_values))

    def get_interpolated_phase(self, abscissa_value): # singular!
        complex_amplitude = self.get_interpolated_complex_amplitude(abscissa_value)
        return numpy.arctan2(numpy.imag(complex_amplitude), numpy.real(complex_amplitude))

    def get_interpolated_phases(self, abscissa_values): # plural!
        complex_amplitudes = self.get_interpolated_complex_amplitudes(abscissa_values)
        return numpy.arctan2(numpy.imag(complex_amplitudes), numpy.real(complex_amplitudes))

    def get_interpolated_intensity(self, abscissa_value):
        return self.get_interpolated_amplitude(abscissa_value)**2

    def get_interpolated_intensities(self, abscissa_values):
        return self.get_interpolated_amplitudes(abscissa_values)**2

    # DEPRECATED METHOD, KEPT FOR RETRO-COMPATIBILITY
    def get_complex_amplitude_from_abscissas(self, abscissa_values):
        warnings.warn("Deprecated function: use get_interpolated_complex_amplitudes", DeprecationWarning)

        return self.electric_field_array.interpolate_values(abscissa_values)

    # modifiers

    def set_wavelength(self,wavelength):
        self.wavelength = wavelength

    def set_wavenumber(self,wavenumber):
        self.wavelength = 2*numpy.pi / wavenumber

    def set_photon_energy(self,photon_energy):
        m2ev = codata.c * codata.h / codata.e      # lambda(m)  = m2eV / energy(eV)
        self.wavelength = m2ev / photon_energy

    def set_complex_amplitude(self,complex_amplitude):
        if complex_amplitude.size != self.electric_field_array.size():
            raise Exception("Complex amplitude array has different dimension")
        self.electric_field_array.np_array = complex_amplitude

    def set_plane_wave_from_complex_amplitude(self, complex_amplitude=(1.0 + 0.0j)):
        self.electric_field_array.np_array = numpy.full(self.electric_field_array.size(), complex_amplitude, dtype=complex)

    def set_plane_wave_from_amplitude_and_phase(self, amplitude=1.0, phase=0.0):
        self.set_plane_wave_from_complex_amplitude(amplitude*numpy.cos(phase) + 1.0j*amplitude*numpy.sin(phase))

    def set_spherical_wave(self, radius=1.0, complex_amplitude=1.0):
        if radius == 0: raise Exception("Radius cannot be zero")
        self.electric_field_array.np_array = complex_amplitude*numpy.exp(1.0j*self.get_wavenumber()*
                                            (self.electric_field_array.scale**2)/(2*radius))

    def add_phase_shift(self, phase_shift):
        self.electric_field_array.np_array *= numpy.exp(1.0j*phase_shift)

    def add_phase_shifts(self, phase_shifts):
        if phase_shifts.size != self.electric_field_array.size():
            raise Exception("Phase Shifts array has different dimension")
        self.electric_field_array.np_array =  numpy.multiply(self.electric_field_array.np_array, numpy.exp(1.0j*phase_shifts))

    def rescale_amplitude(self, factor):
        self.electric_field_array.np_array *= factor

    def rescale_amplitudes(self, factors):
        if factors.size != self.electric_field_array.size(): raise Exception("Factors array has different dimension")
        self.electric_field_array.np_array =  numpy.multiply(self.electric_field_array.np_array, factors)

    def apply_ideal_lens(self, focal_length):
        self.add_phase_shift((-1.0) * self.get_wavenumber() * (self.get_abscissas() ** 2 / focal_length) / 2)

    def apply_slit(self, x_slit_min, x_slit_max):
        window = numpy.ones(self.electric_field_array.size())

        lower_window = numpy.where(self.get_abscissas() < x_slit_min)
        upper_window = numpy.where(self.get_abscissas() > x_slit_max)

        if len(lower_window) > 0: window[lower_window] = 0
        if len(upper_window) > 0: window[upper_window] = 0

        self.rescale_amplitudes(window)

def test_plane_wave(do_plot=0):
    import copy
    #
    # plane wave
    #

    value_amplitude = 5.0
    value_phase = numpy.pi

    wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=1e-2, number_of_points=1000000, x_min=-1, x_max=1)
    wavefront.set_plane_wave_from_amplitude_and_phase(value_amplitude,value_phase)
    wavefront.apply_slit(-0.4, 0.4)

    wavefront_focused = copy.deepcopy(wavefront)
    wavefront_focused.apply_ideal_lens(100)

    #
    # some tests
    #
    test_value1 = wavefront.get_interpolated_amplitude(0.01) - value_amplitude
    assert ( numpy.abs(test_value1) < 1e-6)

    test_value2  = wavefront.get_interpolated_amplitudes(-4)
    assert ( numpy.abs(test_value2) < 1e-6)

    if do_plot:
        from srxraylib.plot.gol import plot

        plot(wavefront.get_abscissas(), wavefront.get_intensity(),show=0,
             title="Plane wave defined in [-1,1] and clipped by a [-.4,.4] slit",
             xtitle="X (m)",ytitle="Intensity")

        plot(wavefront.get_abscissas(), wavefront.get_amplitude(),show=0,
             title="Plane wave defined in [-1,1] and clipped by a [-.4,.4] slit",
             xtitle="X (m)",ytitle="Amplitude")

        plot(wavefront.get_abscissas(), wavefront.get_phase(),show=0,
             title="Plane wave defined in [-1,1] and clipped by a [-.4,.4] slit",
             xtitle="X (m)",ytitle="Phase [rad]")

        plot(wavefront.get_abscissas(), wavefront_focused.get_phase(),show=1,
             title="Plane wave after ideal lens",
             xtitle="X (m)",ytitle="Phase [rad]")


if __name__ == "__main__":

    test_plane_wave(do_plot=1)












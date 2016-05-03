
import numpy
from srxraylib.util.data_structures import ScaledArray, ScaledMatrix

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
    def initialize_wavefront_from_steps(cls, wavelength=1e-10, number_of_points=1000, x_start=0.0, x_step=0.0):
        return Wavefront1D(wavelength, ScaledArray.initialize_from_steps(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         initial_scale_value=x_start,
                                                                         scale_step=x_step))
    @classmethod
    def initialize_wavefront_from_range(cls, wavelength=1e-10, number_of_points=1000, x_min=0.0, x_max=0.0):
        return Wavefront1D(wavelength, ScaledArray.initialize_from_range(np_array=numpy.full(number_of_points, (1.0 + 0.0j), dtype=complex),
                                                                         min_scale_value=x_min,
                                                                         max_scale_value=x_max))
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

    def get_phase(self):
        return numpy.arctan2(numpy.imag(self.get_complex_amplitude()), numpy.real(self.get_complex_amplitude()))

    def get_intensity(self):
        return self.get_amplitude()**2

    def get_complex_amplitude_from_abscissa(self, abscissa_value):
        return self.electric_field_array.interpolate_value(abscissa_value)

    def get_complex_amplitude_from_abscissas(self, abscissa_values):
        return self.electric_field_array.interpolate_values(abscissa_values)

    def get_amplitude_from_abscissa(self, abscissa_value):
        return numpy.absolute(self.get_complex_amplitude_from_abscissa(abscissa_value))

    def get_amplitude_from_abscissas(self, abscissa_values):
        return numpy.absolute(self.get_complex_amplitude_from_abscissas(abscissa_values))

    def get_phase_from_abscissa(self, abscissa_value):
        complex_amplitude = self.get_complex_amplitude_from_abscissa(abscissa_value)
        return numpy.arctan2(numpy.imag(complex_amplitude), numpy.real(complex_amplitude))

    def get_phase_from_abscissas(self, abscissa_values):
        complex_amplitudes = self.get_complex_amplitude_from_abscissas(abscissa_values)
        return numpy.arctan2(numpy.imag(complex_amplitudes), numpy.real(complex_amplitudes))

    def get_intensity_from_abscissa(self, abscissa_value):
        return self.get_amplitude_from_abscissa(abscissa_value)**2

    def get_intensity_from_abscissas(self, abscissa_values):
        return self.get_amplitude_from_abscissas(abscissa_values)**2

    def set_plane_wave_from_complex_amplitude(self, complex_amplitude=(1.0 + 0.0j)):
        self.electric_field_array.np_array = numpy.full(self.electric_field_array.size(), complex_amplitude, dtype=complex)

    def set_plane_wave_from_amplitude_and_phase(self, amplitude=1.0, phase=0.0):
        self.set_plane_wave_from_complex_amplitude(amplitude*numpy.cos(phase) + 1.0j*amplitude*numpy.sin(phase))

    def set_spherical_wave(self, amplitude=1.0, radius=1.0):
        if radius == 0: raise Exception("Radius cannot be zero")
        self.electric_field_array.np_array = (amplitude/radius)*numpy.exp(-1.0j*self.get_wavenumber()*(self.electric_field_array.scale**2)/(2*radius))

    def add_phase_shift(self, phase_shift):
        self.electric_field_array.np_array *= numpy.exp(1.0j*phase_shift)

    def add_phase_shifts(self, phase_shifts):
        if phase_shifts.size != self.electric_field_array.size(): raise Exception("Phase Shifts array has different dimension")
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

if __name__ == "__main__":

    wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=1e-2, number_of_points=1000000, x_min=-1, x_max=1)
    wavefront.set_plane_wave_from_amplitude_and_phase(2, 1.0)
    wavefront.apply_slit(-0.4, 0.4)
    wavefront.apply_ideal_lens(100)

    import matplotlib.pylab as plt
    f1 = plt.figure(1)
    plt.plot(wavefront.get_abscissas(), wavefront.get_intensity())
    plt.title("Manolo for President")
    plt.xlabel("X (m)")
    plt.ylabel("I V^2 m^-2")
    plt.show()










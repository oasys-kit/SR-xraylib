import numpy

from srxraylib.util.data_structures import ScaledArray
from srxraylib.waveoptics.wavefront import Wavefront1D

def propagate_1D_fresnel(wavefront, propagation_distance):
    fft_scale = numpy.fft.fftfreq(wavefront.size())/wavefront.delta()

    fft = numpy.fft.fft(wavefront.get_complex_amplitude())
    fft *= numpy.exp((-1.0j) * numpy.pi * wavefront.get_wavelength() * propagation_distance * fft_scale**2)
    ifft = numpy.fft.ifft(fft)

    return Wavefront1D(wavefront.get_wavelength(), ScaledArray.initialize_from_steps(ifft, wavefront.offset(), wavefront.delta()))


if __name__ == "__main__":
    wavelength = 1.24e-10 # 10keV
    aperture_diameter = 40e-6 # 1e-3 # 1e-6
    detector_size = 800e-6
    distance = 3.6
    npoints = 1000

    wavefront = Wavefront1D.initialize_wavefront_from_range(wavelength=wavelength, number_of_points=npoints, x_min=-detector_size/2, x_max=detector_size/2)
    wavefront.set_plane_wave_from_complex_amplitude((2.0+1.0j))
    wavefront.apply_slit(-aperture_diameter/2, aperture_diameter/2)

    wavefront_2 = propagate_1D_fresnel(wavefront, distance)

    wavefront.apply_ideal_lens(distance)
    wavefront_3 = propagate_1D_fresnel(wavefront, distance)


    import matplotlib.pylab as plt
    f1 = plt.figure(1)
    plt.plot(wavefront_2.get_abscissas()*1e6, wavefront_2.get_intensity())
    plt.plot(wavefront_3.get_abscissas()*1e6, wavefront_3.get_intensity())
    plt.title("Manolo for President")
    plt.xlabel("X (m)")
    plt.ylabel("I V^2 m^-2")
    plt.xlim([-60, 60])
    plt.show()



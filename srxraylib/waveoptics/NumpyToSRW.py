import numpy
from oasys_srw.srwlib import *
import numpy as np
from srxraylib import deprecated


@deprecated("use wofrysrw, instead")
def numpyArrayToSRWArray(numpy_array):
    """
    Converts a numpy.array to an array usable by SRW.
    :param numpy_array: a 2D numpy array
    :return: a 2D complex SRW array
    """
    elements_size = numpy_array.size

    r_horizontal_field = numpy_array[:, :].real.transpose().flatten().astype(np.float)
    i_horizontal_field = numpy_array[:, :].imag.transpose().flatten().astype(np.float)

    tmp = np.zeros(elements_size * 2, dtype=np.float32)
    for i in range(elements_size):
        tmp[2*i] = r_horizontal_field[i]
        tmp[2*i+1] = i_horizontal_field[i]

    return array('f', tmp)

@deprecated("use wofrysrw, instead")
def SRWWavefrontFromElectricField(horizontal_start, horizontal_end, horizontal_efield,
                                  vertical_start, vertical_end, vertical_efield,
                                  energy, z, Rx, dRx, Ry, dRy):
    """
    Creates a SRWWavefront from pi and sigma components of the electrical field.
    :param horizontal_start: Horizontal start position of the grid in m
    :param horizontal_end: Horizontal end position of the grid in m
    :param horizontal_efield: The pi component of the complex electrical field
    :param vertical_start: Vertical start position of the grid in m
    :param vertical_end: Vertical end position of the grid in m
    :param vertical_efield: The sigma component of the complex electrical field
    :param energy: Energy in eV
    :param z: z position of the wavefront in m
    :param Rx: Instantaneous horizontal wavefront radius
    :param dRx: Error in instantaneous horizontal wavefront radius
    :param Ry: Instantaneous vertical wavefront radius
    :param dRy: Error in instantaneous vertical wavefront radius
    :return: A wavefront usable with SRW.
    """

    horizontal_size = horizontal_efield.shape[0]
    vertical_size = horizontal_efield.shape[1]

    if horizontal_size % 2 == 1 or \
       vertical_size % 2 == 1:
        # raise Exception("Both horizontal and vertical grid must have even number of points")
        print("NumpyToSRW: WARNING: Both horizontal and vertical grid must have even number of points")

    horizontal_field = numpyArrayToSRWArray(horizontal_efield)
    vertical_field = numpyArrayToSRWArray(vertical_efield)

    srw_wavefront = SRWLWfr(_arEx=horizontal_field,
                            _arEy=vertical_field,
                            _typeE='f',
                            _eStart=energy,
                            _eFin=energy,
                            _ne=1,
                            _xStart=horizontal_start,
                            _xFin=horizontal_end,
                            _nx=horizontal_size,
                            _yStart=vertical_start,
                            _yFin=vertical_end,
                            _ny=vertical_size,
                            _zStart=z)

    srw_wavefront.Rx = Rx
    srw_wavefront.Ry = Ry
    srw_wavefront.dRx = dRx
    srw_wavefront.dRy = dRy

    return srw_wavefront

@deprecated("use wofrysrw, instead")
def SRWArrayToNumpy(srw_array, dim_x, dim_y, number_energies):
    """
    Converts a SRW array to a numpy.array.
    :param srw_array: SRW array
    :param dim_x: size of horizontal dimension
    :param dim_y: size of vertical dimension
    :param number_energies: Size of energy dimension
    :return: 4D numpy array: [energy, horizontal, vertical, polarisation={0:horizontal, 1: vertical}]
    """
    re = np.array(srw_array[::2], dtype=np.float)
    im = np.array(srw_array[1::2], dtype=np.float)

    e = re + 1j * im
    e = e.reshape((dim_y,
                   dim_x,
                   number_energies,
                   1)
                  )

    e = e.swapaxes(0, 2)

    return e.copy()

@deprecated("use wofrysrw, instead")
def SRWEFieldAsNumpy(srw_wavefront):
    """
    Extracts electrical field from a SRWWavefront
    :param srw_wavefront: SRWWavefront to extract electrical field from.
    :return: 4D numpy array: [energy, horizontal, vertical, polarisation={0:horizontal, 1: vertical}]
    """

    dim_x = srw_wavefront.mesh.nx
    dim_y = srw_wavefront.mesh.ny
    number_energies = srw_wavefront.mesh.ne

    x_polarization = SRWArrayToNumpy(srw_wavefront.arEx, dim_x, dim_y, number_energies)
    y_polarization = SRWArrayToNumpy(srw_wavefront.arEy, dim_x, dim_y, number_energies)

    e_field = np.concatenate((x_polarization,y_polarization), 3)

    return e_field

if __name__=="__main__":
    a = numpyArrayToSRWArray(numpy.zeros((10, 2)))
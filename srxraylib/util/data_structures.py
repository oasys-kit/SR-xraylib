import copy
import numpy
import scipy

#######################################
#
# UTILITY IGOR-LIKE VECTORIAL CLASS
# AND FUNCTIONS
#
#######################################

class ScaledMatrix(object):
    x_coord = None
    y_coord = None
    z_values = None

    def __init__(self, x_coord=numpy.zeros(0), y_coord=numpy.zeros(0), z_values=numpy.zeros((0, 0)), interpolator=False):
        if z_values.shape != (len(x_coord), len(y_coord)): raise Exception("z_values shape " + str(z_values.shape) + " != " + str((len(x_coord), len(y_coord))))

        self.x_coord = numpy.round(x_coord, 12)
        self.y_coord = numpy.round(y_coord, 12)
        self.z_values = numpy.round(z_values, 12)

        self.stored_shape = self._shape()

        if interpolator:
            self.interpolator = scipy.interpolate.RectBivariateSpline(self.x_coords, self.y_coords, self.z_values)

    def get_x_value(self, index):
        return self.x_coord[index]

    def get_y_value(self, index):
        return self.y_coord[index]

    def _shape(self):
        return (len(self.x_coord), len(self.y_coord))

    def shape(self):
        return self.stored_shape

    def get_z_value(self, x_index, y_index):
        return self.z_values[x_index][y_index]

    def set_z_value(self, x_index, y_index, z_value):
        self.z_values[x_index][y_index] = z_value

    def interpolate_value(self, x_coord, y_coord):
        if self.interpolator == None: raise Exception("Scaled Matrix not initialized with interpolator")

        return self.interpolator.ev(x_coord, y_coord)

    '''
    Equivalent to the IGOR command: SetScale /P (wave, min value, max value)
    '''
    def set_scale_from_steps(self, axis, initial_scale_value, scale_step):
        if self.stored_shape()[axis] > 0:
            if axis < 0 or axis > 1: raise Exception("Axis must be 0 or 1, found: " + str(axis))
            if scale_step <= 0.0: raise Exception("Scale Step must be > 0.0")

            # Problem in comparison between float64 and numpy.float64:
            # reduce precision to avoid crazy research results

            scale = numpy.round(initial_scale_value, 12) + numpy.arange(0, len(self.stored_shape[axis])) * numpy.round(scale_step, 12)

            if axis == 0: self.x_coord = scale
            elif axis == 1: self.y_coord = scale

            self.stored_shape = self._shape()

    '''
    Equivalent to the IGOR command: SetScale /I (wave, min value, max value)
    '''
    def set_scale_from_range(self, axis, min_scale_value, max_scale_value):
        if self.stored_shape()[axis] > 0:
            if axis < 0 or axis > 1: raise Exception("Axis must be 0 or 1, found: " + str(axis))
            if max_scale_value <= min_scale_value: raise Exception("Max Scale Value ("+ str(max_scale_value) + ") must be > Min Scale Vale(" + str(min_scale_value) + ")")

            self.set_scale_from_steps(axis, min_scale_value, (max_scale_value - min_scale_value)/(len(self.stored_shape[axis])-1))

#######################################

class ScaledArray(object):
    def __init__(self, np_array = numpy.zeros(0), scale = numpy.zeros(0)):
        if len(np_array) != len(scale): raise Exception("np_array and scale must have the same dimension: (" + str(len(np_array)) + " != " + str(len(scale)) )
        if len(np_array) == 0: raise Exception("np_array can't have 0 size")

        # Problem in comparison between float64 and numpy.float64:
        # reduce precision to avoid crazy research results

        self.np_array = numpy.round(np_array, 12)
        self.scale = numpy.round(scale, 12)

        self.stored_delta = self._delta()
        self.stored_offset = self._offset()
        self.stored_size = self._size()

        self._v_interpolate_values = numpy.vectorize(self.interpolate_value)

    @classmethod
    def initialize(cls, np_array = numpy.zeros(0)):
        return ScaledArray(np_array, numpy.arange(0, len(np_array)))

    @classmethod
    def initialize_from_range(cls, np_array , min_scale_value, max_scale_value):
        array = ScaledArray.initialize(np_array)
        array.set_scale_from_range(min_scale_value, max_scale_value)
        return array

    @classmethod
    def initialize_from_steps(cls, np_array, initial_scale_value, scale_step):
        array = ScaledArray.initialize(np_array)
        array.set_scale_from_steps(initial_scale_value, scale_step)
        return array

    def _size(self):
        if len(self.np_array) != len(self.scale): raise Exception("np_array and scale must have the same dimension")

        return len(self.np_array)

    def _offset(self):
        if len(self.scale) > 1:
            return self.scale[0]
        else:
            return numpy.nan

    def _delta(self):
        if len(self.scale) > 1:
            return abs(self.scale[1]-self.scale[0])
        else:
            return 0.0

    def size(self):
        return self.stored_size

    def offset(self):
        return self.stored_offset

    def delta(self):
        return self.stored_delta

    def get_scale_value(self, index):
        return self.scale[index]

    def get_value(self, index):
        return self.np_array[index]

    def set_value(self, index, value):
        self.np_array[index] = value

    def interpolate_values(self, scale_values):
        return self._v_interpolate_values(scale_values)

    def interpolate_value(self, scale_value):
        scale_value = numpy.round(scale_value, 12)
        scale_0 = self.scale[0]

        if scale_value <= scale_0: return self.np_array[0]
        if scale_value >= self.scale[self.size()-1]: return self.np_array[self.size()-1]

        approximated_index = (scale_value-scale_0)/self.stored_delta

        index_0 = int(numpy.floor(approximated_index))
        index_1 = int(numpy.ceil(approximated_index))

        x_0 = self.scale[index_0]
        x_1 = self.scale[index_1]
        y_0 = self.np_array[index_0]
        y_1 = self.np_array[index_1]

        return y_0 + ((y_1 - y_0) * (scale_value - x_0)/(x_1 - x_0))

    def interpolate_scale_value(self, value): # only for monotonic np_array
        return numpy.interp(value, self.np_array, self.scale)

    '''
    Equivalent to the IGOR command: SetScale /P (wave, offset, step)
    '''
    def set_scale_from_steps(self, initial_scale_value, scale_step):
        if self.size() > 0:
            if scale_step <= 0.0: raise Exception("Scale Step must be > 0.0")

            # Problem in comparison between float64 and numpy.float64:
            # reduce precision to avoid crazy research results

            self.scale = numpy.round(initial_scale_value, 12) + numpy.arange(0, len(self.np_array)) * numpy.round(scale_step, 12)

            self.stored_offset = self._offset()
            self.stored_delta = self._delta()

    '''
    Equivalent to the IGOR command: SetScale /I (wave, min value, max value)
    '''
    def set_scale_from_range(self, min_scale_value, max_scale_value):
        if max_scale_value <= min_scale_value: raise Exception("Max Scale Value ("+ str(max_scale_value) + ") must be > Min Scale Vale(" + str(min_scale_value) + ")")

        self.set_scale_from_steps(min_scale_value, (max_scale_value - min_scale_value)/(len(self.np_array)-1))

#######################################
#######################################
#######################################

if __name__ == "__main__":
     scaled_array = ScaledArray.initialize(numpy.arange(15.0, 18.8, 0.2))
     scaled_array.set_scale_from_steps(15.0, 0.2)

     print("interp", scaled_array.interpolate_value(18.8))
     print("interp", scaled_array.interpolate_value(16.22))
     print("interp",  scaled_array.interpolate_value(22.35))

     scaled_array = ScaledArray.initialize(numpy.arange(15.0, 18.8, 0.2))
     scaled_array.set_scale_from_range(15.0, 18.8)

     print("interp", scaled_array.interpolate_value(18.8))
     print("interp", scaled_array.interpolate_value(16.22))
     print("interp",  scaled_array.interpolate_value(22.35))

     v_interpolate = numpy.vectorize(scaled_array.interpolate_value)
     values = v_interpolate([10, 17.3, 16.5, 20, 30])

     print(values)

     scaled_array = ScaledArray.initialize_from_steps(numpy.arange(15.0, 18.8, 0.2), 15.0, 0.2)

     print("interp", scaled_array.interpolate_value(18.8))
     print("interp", scaled_array.interpolate_value(16.22))
     print("interp",  scaled_array.interpolate_value(22.35))

     scaled_array = ScaledArray.initialize_from_range(numpy.arange(15.0, 18.8, 0.2),15.0, 18.8)

     print("interp", scaled_array.interpolate_value(18.8))
     print("interp", scaled_array.interpolate_value(16.22))
     print("interp",  scaled_array.interpolate_value(22.35))

     print("interp", scaled_array.interpolate_scale_value(18.8))
     print("interp", scaled_array.interpolate_scale_value(16.22))




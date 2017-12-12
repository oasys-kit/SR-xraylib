
import numpy


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
        self._set_is_complex_matrix()

        self.stored_shape = self._shape()

        self.stored_delta_x = self._delta_x()
        self.stored_offset_x = self._offset_x()
        self.stored_size_x = self._size_x()

        self.stored_delta_y = self._delta_y()
        self.stored_offset_y = self._offset_y()
        self.stored_size_y = self._size_y()

        #
        # write now interpolator=True means that the interpolator is up to date and stored in the
        # interpolated value. interpolator must be changed to False when something of the wavefront
        # is changed. When interpolation is required, compute_interpolator() is called.
        #
        # Withouth loss of generality, Wavefront2D can always be initialize with
        # interpolator=False. Then it will be switched on when needed.
        # It has been implemented for saving time: the interpolator is recomputed only if needed

        # DANGER: if  self.x_coord, self.y_coord or self.z_values are changed directly by the user
        # (without using set* methods), then set set.interpolator=False must be manually set,
        self.interpolator = interpolator
        self.interpolator_value = None

        if interpolator:
            self.compute_interpolator()

    @classmethod
    def initialize(cls, np_array_z = numpy.zeros((0,0)),interpolator=False):
        return ScaledMatrix(numpy.zeros(np_array_z.shape[0]),
                            numpy.zeros(np_array_z.shape[1]),
                            np_array_z,interpolator=interpolator)
    @classmethod
    def initialize_from_range(cls, np_array ,
                              min_scale_value_x, max_scale_value_x,
                              min_scale_value_y, max_scale_value_y,
                              interpolator=False):
        array = ScaledMatrix.initialize(np_array)
        array.set_scale_from_range(0,min_scale_value_x, max_scale_value_x)
        array.set_scale_from_range(1,min_scale_value_y, max_scale_value_y)
        if interpolator:
            array.compute_interpolator()
        return array

    @classmethod
    def initialize_from_steps(cls, np_array, initial_scale_value_x, scale_step_x, initial_scale_value_y, scale_step_y,
                              interpolator=False):
        array = ScaledMatrix.initialize(np_array)
        array.set_scale_from_steps(0,initial_scale_value_x, scale_step_x)
        array.set_scale_from_steps(1,initial_scale_value_y, scale_step_y)
        if interpolator:
            array.compute_interpolator()
        return array

    def _size_x(self):
        return len(self.x_coord)

    def _offset_x(self):
        if len(self.x_coord) > 1:
            return self.x_coord[0]
        else:
            return numpy.nan

    def _delta_x(self):
        if len(self.x_coord) > 1:
            return abs(self.x_coord[1]-self.x_coord[0])
        else:
            return 0.0

    def size_x(self):
        return self.stored_size_x

    def offset_x(self):
        return self.stored_offset_x

    def delta_x(self):
        return self.stored_delta_x

    def _size_y(self):
        return len(self.y_coord)

    def _offset_y(self):
        if len(self.y_coord) > 1:
            return self.y_coord[0]
        else:
            return numpy.nan

    def _delta_y(self):
        if len(self.y_coord) > 1:
            return abs(self.y_coord[1]-self.y_coord[0])
        else:
            return 0.0

    def size_y(self):
        return self.stored_size_y

    def offset_y(self):
        return self.stored_offset_y

    def delta_y(self):
        return self.stored_delta_y

    def get_x_value(self, index):
        return self.x_coord[index]

    def get_y_value(self, index):
        return self.y_coord[index]

    def get_x_values(self): # plural!!
        return self.x_coord

    def get_y_values(self): # plural!!
        return self.y_coord

    def _shape(self):
        return (len(self.x_coord), len(self.y_coord))

    def shape(self):
        return self.stored_shape

    def size(self):
        return self.stored_shape

    def get_z_value(self, x_index, y_index):
        return self.z_values[x_index][y_index]

    def get_z_values(self):  # plural!
        return self.z_values

    def set_z_value(self, x_index, y_index, z_value):
        self.z_values[x_index][y_index] = z_value
        if isinstance(z_value, complex): self._is_complex_matrix = True
        self.interpolator = False

    def set_z_values(self, new_value):
        if new_value.shape != self.shape():
            raise Exception("New data set must have same shape as old one")
        else:
            self.z_values = new_value
            self._set_is_complex_matrix()
            self.interpolator = False

    def _set_is_complex_matrix(self):
        self._is_complex_matrix = True in numpy.iscomplex(self.z_values)

    def is_complex_matrix(self):
        return self._is_complex_matrix

    def interpolate_value(self, x_coord, y_coord):
        if self.interpolator == False:
            self.compute_interpolator()
        if self.is_complex_matrix():
            return self.interpolator_value[0].ev(x_coord, y_coord) + 1j * self.interpolator_value[1].ev(x_coord, y_coord)
        else:
            return self.interpolator_value.ev(x_coord, y_coord)

    def compute_interpolator(self):
        from scipy import interpolate
        print("ScaledMatrix.compute_interpolator: Computing interpolator...")

        if self.is_complex_matrix():
            self.interpolator_value = (
                interpolate.RectBivariateSpline(self.x_coord, self.y_coord, numpy.real(self.z_values)),
                interpolate.RectBivariateSpline(self.x_coord, self.y_coord, numpy.imag(self.z_values)),
                )
        else:
            self.interpolator_value = interpolate.RectBivariateSpline(self.x_coord, self.y_coord, self.z_values)

        self.interpolator = True

    '''
    Equivalent to the IGOR command: SetScale /P (wave, min value, max value)
    '''
    def set_scale_from_steps(self, axis, initial_scale_value, scale_step):
        if self.stored_shape[axis] > 0:
            if axis < 0 or axis > 1: raise Exception("Axis must be 0 or 1, found: " + str(axis))
            if scale_step <= 0.0: raise Exception("Scale Step must be > 0.0")

            # Problem in comparison between float64 and numpy.float64:
            # reduce precision to avoid crazy research results

            scale = numpy.round(initial_scale_value, 12) + numpy.arange(0, (self.stored_shape[axis])) * numpy.round(scale_step, 12)
            if axis == 0:
                self.x_coord = scale
                self.stored_delta_x = self._delta_x()
                self.stored_offset_x = self._offset_x()
            elif axis == 1:
                self.y_coord = scale
                self.stored_delta_y = self._delta_y()
                self.stored_offset_y = self._offset_y()

            self.stored_shape = self._shape()
            self.interpolator = False

    '''
    Equivalent to the IGOR command: SetScale /I (wave, min value, max value)
    '''
    def set_scale_from_range(self, axis, min_scale_value, max_scale_value):
        if self.stored_shape[axis] > 0:
            if axis < 0 or axis > 1: raise Exception("Axis must be 0 or 1, found: " + str(axis))
            if max_scale_value <= min_scale_value: raise Exception("Max Scale Value ("+ str(max_scale_value) + ") must be > Min Scale Vale(" + str(min_scale_value) + ")")

            self.set_scale_from_steps(axis, min_scale_value, (max_scale_value - min_scale_value)/((self.stored_shape[axis])-1))
            self.interpolator = False

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

    def get_values(self):  # Plural!
        return self.np_array

    def get_abscissas(self):
        return self.scale # numpy.linspace(self.offset(),self.offset()+self.delta()*(self.size()-1),self.size())

    def set_value(self, index, value):
        self.np_array[index] = value

    def set_values(self, value):
        if self.size() != value.size:
            raise Exception("Incompatible dimensions")
        self.np_array = value

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

        if x_0 == x_1:
            return y_0
        else:
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





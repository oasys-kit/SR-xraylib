"""
Utility to mimic IGOR-like vector and matrix classes and functions.
"""
import numpy

class ScaledMatrix(object):
    """
    ScaledMatrix stores a 2D array and its axes.
    Constructor.

    Parameters
    ----------
    x_coord : numpy array
        the array for X.
    y_coord : numpy array
        the array for Y.
    z_values : numpy array
        the 2D array for Z.
    interpolator : boolean, optional
        True means that the interpolator has been calculates and it is ready. False means that is not ready.
        interpolator=True means that the interpolator is up to date and stored in the interpolated value.
        interpolator must be changed to False when something of the data is changed.
        When interpolation is required, compute_interpolator() is called.
        Withouth loss of generality, Wavefront2D can always be initialize with
        interpolator=False. Then it will be switched on when needed.
        It has been implemented for saving time: the interpolator is recomputed only if needed.

    """
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
    def initialize(cls, np_array_z = numpy.zeros((0,0)), interpolator=False):
        """
        Initialize a ScaledMatrix instance from a 2D array.
        Parameters
        ----------
        np_array_z : numpy array
            The 2D array.
        interpolator : boolean, optional
            True means that the interpolator has been calculates and it is ready. False means that is not ready.

        Returns
        -------
        ScaledMatrix instance
        """
        return ScaledMatrix(numpy.zeros(np_array_z.shape[0]),
                            numpy.zeros(np_array_z.shape[1]),
                            np_array_z,interpolator=interpolator)
    @classmethod
    def initialize_from_range(cls, np_array,
                              min_scale_value_x, max_scale_value_x,
                              min_scale_value_y, max_scale_value_y,
                              interpolator=False):
        """
        Initializes a ScaledMatrix instance from a 2D array and the range of the axes.

        Parameters
        ----------
        np_array : numpy array
            The 2D array.
        min_scale_value_x : float
        max_scale_value_x : float
        min_scale_value_y : float
        max_scale_value_y : float
        interpolator : boolean, optional
            True means that the interpolator has been calculates and it is ready. False means that is not ready.

        Returns
        -------
        ScaledMatrix instance
        """
        array = ScaledMatrix.initialize(np_array)
        array.set_scale_from_range(0,min_scale_value_x, max_scale_value_x)
        array.set_scale_from_range(1,min_scale_value_y, max_scale_value_y)
        if interpolator:
            array.compute_interpolator()
        return array

    @classmethod
    def initialize_from_steps(cls, np_array, initial_scale_value_x, scale_step_x, initial_scale_value_y, scale_step_y,
                              interpolator=False):
        """
        Initializes a ScaledMatrix instance from a 2D array and the steps of the axes.

        Parameters
        ----------
        np_array : numpy array
            The 2D array.
        initial_scale_value_x : float
        scale_step_x : float
        initial_scale_value_y : float
        scale_step_y : float
        interpolator : boolean, optional
            True means that the interpolator has been calculates and it is ready. False means that is not ready.

        Returns
        -------
        ScaledMatrix instance
        """
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
        """
        Returns the size of the X axis.

        Returns
        -------
        int
        """
        return self.stored_size_x

    def offset_x(self):
        """
        returns the offset of the X axis.

        Returns
        -------
        float
        """
        return self.stored_offset_x

    def delta_x(self):
        """
        Returns the delta (step) of the X axis.

        Returns
        -------
        int
        """
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
        """
        Returns the size of the Y axis.

        Returns
        -------
        int
        """
        return self.stored_size_y

    def offset_y(self):
        """
        Returns the offset of the Y axis.

        Returns
        -------
        float
        """
        return self.stored_offset_y

    def delta_y(self):
        """
        Returns the delta (step) of the Y axis.

        Returns
        -------
        float
        """
        return self.stored_delta_y

    def get_x_value(self, index):
        """
        Returns the X value for a given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        float
        """
        return self.x_coord[index]

    def get_y_value(self, index):
        """
        Returns the Y value for a given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        float
        """
        return self.y_coord[index]

    def get_x_values(self): # plural!!
        """
        Returns the X array.

        Returns
        -------
        numpy array
        """
        return self.x_coord

    def get_y_values(self): # plural!!
        """
        Returns the Y array.

        Returns
        -------
        numpy array
        """
        return self.y_coord

    def _shape(self):
        return (len(self.x_coord), len(self.y_coord))

    def shape(self):
        """
        Returns the shape.

        Returns
        -------
        tuple
        """
        return self.stored_shape

    def size(self):
        """
        Returns the size of the array.

        Returns
        -------
        int
        """
        return self.stored_shape

    def get_z_value(self, x_index, y_index):
        """
        Returns the Z (2D array) value for given X,Y indices.

        Parameters
        ----------
        x_index : int
            The index for axis 0.
        y_index : int
            The index for axis 1.

        Returns
        -------
        float
        """
        return self.z_values[x_index][y_index]

    def get_z_values(self):  # plural!
        """
        Returns the Z (2D array) array.

        Returns
        -------
        numpy array
        """
        return self.z_values

    def set_z_value(self, x_index, y_index, z_value):
        """
        Sets a given Z value at given indices.

        Parameters
        ----------
        x_index : int
            The index for axis 0.
        y_index : int
            The index for axis 1.
        z_value : float
            The value to be stored.
        """
        self.z_values[x_index][y_index] = z_value
        if isinstance(z_value, complex): self._is_complex_matrix = True
        self.interpolator = False

    def set_z_values(self, new_value):
        """
        Sets a given 2D array.

        Parameters
        ----------
        new_value : numpy array
            The 2D array.
        """
        if new_value.shape != self.shape():
            raise Exception("New data set must have same shape as old one")
        else:
            self.z_values = new_value
            self._set_is_complex_matrix()
            self.interpolator = False

    def _set_is_complex_matrix(self):
        self._is_complex_matrix = True in numpy.iscomplex(self.z_values)

    def is_complex_matrix(self):
        """
        Returns True if the data stored is of complex type.

        Returns
        -------
        boolean
        """
        return self._is_complex_matrix

    def interpolate_value(self, x_coord, y_coord):
        """
        Gives the Z value interpolated at given coordinates (x_coord, y_coord).

        Parameters
        ----------
        x_coord : float
            The coordinate at axis 0.
        y_coord : float
            The coordinate at axis 1.

        Returns
        -------
        float
        """
        if self.interpolator == False:
            self.compute_interpolator()
        if self.is_complex_matrix():
            return self.interpolator_value[0].ev(x_coord, y_coord) + 1j * self.interpolator_value[1].ev(x_coord, y_coord)
        else:
            return self.interpolator_value.ev(x_coord, y_coord)

    def compute_interpolator(self):
        """
        Calculates and stores the interpolation object.
        """
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


    def set_scale_from_steps(self, axis, initial_scale_value, scale_step):
        """
        Equivalent to the IGOR command: SetScale /P (wave, min value, max value).

        Parameters
        ----------
        axis : int
        initial_scale_value : float
        scale_step : float
        """
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

    def set_scale_from_range(self, axis, min_scale_value, max_scale_value):
        """
        Equivalent to the IGOR command: SetScale /I (wave, min value, max value).

        Parameters
        ----------
        axis : int
        min_scale_value : float
        max_scale_value : float
        """
        if self.stored_shape[axis] > 0:
            if axis < 0 or axis > 1: raise Exception("Axis must be 0 or 1, found: " + str(axis))
            if max_scale_value <= min_scale_value: raise Exception("Max Scale Value ("+ str(max_scale_value) + ") must be > Min Scale Vale(" + str(min_scale_value) + ")")

            self.set_scale_from_steps(axis, min_scale_value, (max_scale_value - min_scale_value)/((self.stored_shape[axis])-1))
            self.interpolator = False

#######################################

class ScaledArray(object):
    """
    Stores a 1D array and the abscissas information.
    Constructor.

    Parameters
    ----------
    np_array : numpy array
        The array to be stored.
    scale : float
        The scale values (np_array and scale must have the same dimension).
    """
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
        """
        Initializes a ScaledArray instance.

        Parameters
        ----------
        np_array : numpy array
                    The array to be stored.

        Returns
        -------
        ScaledArray instance
        """
        return ScaledArray(np_array, numpy.arange(0, len(np_array)))

    @classmethod
    def initialize_from_range(cls, np_array , min_scale_value, max_scale_value):
        """

        Parameters
        ----------
        np_array: numpy array
                    The array to be stored.
        min_scale_value : float
        max_scale_value : float

        Returns
        -------
        ScaledArray instance
        """
        array = ScaledArray.initialize(np_array)
        array.set_scale_from_range(min_scale_value, max_scale_value)
        return array

    @classmethod
    def initialize_from_steps(cls, np_array, initial_scale_value, scale_step):
        """

        Parameters
        ----------
        np_array: numpy array
                    The array to be stored.
        initial_scale_value : float
        scale_step : float

        Returns
        -------
        A ScaledArray instance.
        """
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
        """
        Returns the size of the array.

        Returns
        -------
        int
        """
        return self.stored_size

    def offset(self):
        """
        Returns the offset of the array.

        Returns
        -------
        float
        """
        return self.stored_offset

    def delta(self):
        """
        Returns the delta (step) of the array.

        Returns
        -------
        float
        """
        return self.stored_delta

    def get_scale_value(self, index):
        """
        Returns the scale at a particular index.

        Parameters
        ----------
        index : int

        Returns
        -------
        float
        """
        return self.scale[index]

    def get_value(self, index):
        """
        Gets the value for a given index.

        Parameters
        ----------
        index : int

        Returns
        -------
        float
        """
        return self.np_array[index]

    def get_values(self):  # Plural!
        """
        Returns the array.

        Returns
        -------
        numpy array
        """
        return self.np_array

    def get_abscissas(self):
        """
        Return an array with the abscissas.

        Returns
        -------
        numpy array
        """
        return self.scale # numpy.linspace(self.offset(),self.offset()+self.delta()*(self.size()-1),self.size())

    def set_value(self, index, value):
        """
        Sets a value at a particular index.

        Parameters
        ----------
        index : int
        value : float
        """
        self.np_array[index] = value

    def set_values(self, value):
        """
        Sets an array.

        Parameters
        ----------
        value : numpy array
        """
        if self.size() != value.size:
            raise Exception("Incompatible dimensions")
        self.np_array = value

    def interpolate_values(self, scale_values):
        """
        Get the interpolated values for given scale or abscissas values.

        Parameters
        ----------
        scale_values : numpy array

        Returns
        -------
        numpy array
        """
        return self._v_interpolate_values(scale_values)

    def interpolate_value(self, scale_value):
        """
        Get the interpolated value for a given scale or abscissas value.

        Parameters
        ----------
        scale_value : int

        Returns
        -------
        float
        """
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
        """
        Interpolate only for monotonic np_array.

        Parameters
        ----------
        value : float

        Returns
        -------
        float
        """
        return numpy.interp(value, self.np_array, self.scale)

    def set_scale_from_steps(self, initial_scale_value, scale_step):
        """
        Equivalent to the IGOR command: SetScale /P (wave, offset, step).

        Parameters
        ----------
        initial_scale_value : float
        scale_step : float
        """
        if self.size() > 0:
            # Problem in comparison between float64 and numpy.float64:
            # reduce precision to avoid crazy research results

            self.scale = numpy.round(initial_scale_value, 12) + numpy.arange(0, len(self.np_array)) * numpy.round(scale_step, 12)

            self.stored_offset = self._offset()
            self.stored_delta = self._delta()

    def set_scale_from_range(self, min_scale_value, max_scale_value):
        """
        Equivalent to the IGOR command: SetScale /I (wave, min value, max value).

        Parameters
        ----------
        min_scale_value : float
        max_scale_value : float
        """
        if max_scale_value <= min_scale_value: raise Exception("Max Scale Value ("+ str(max_scale_value) + ") must be > Min Scale Vale(" + str(min_scale_value) + ")")

        self.set_scale_from_steps(min_scale_value, (max_scale_value - min_scale_value)/(len(self.np_array)-1))






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

        self.stored_shape = self._shape()

        #
        # write now interpolator means that the interpolator is up to date and stored in the
        # interpolated value. interpolator must be changed to False when something of the wavefront
        # is changed. When interpolation is requiref, compute_interpolator() is called.
        # Withouth loss of generality, Wavefront2D can always be initialize with
        # interpolator=False. Then it will be switched on when needed.
        # It has been implemented for saving time in recomputing the interpolator if it is not needed
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
        self.interpolator = False

    def set_z_values(self, new_value):
        if new_value.shape != self.shape():
            raise Exception("New data set must have same shape as old one")
        else:
            self.z_values = new_value
            self.interpolator = False

    def interpolate_value(self, x_coord, y_coord):
        if self.interpolator == False:
            self.compute_interpolator()
        return self.interpolator_value[0].ev(x_coord, y_coord) + 1j * self.interpolator_value[1].ev(x_coord, y_coord)

    def compute_interpolator(self):
        from scipy import interpolate
        print("<><><><><> Computing interpolator...")
        self.interpolator_value = (
            interpolate.RectBivariateSpline(self.x_coord, self.y_coord, numpy.real(self.z_values)),
            interpolate.RectBivariateSpline(self.x_coord, self.y_coord, numpy.imag(self.z_values)),
            )
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
            if axis == 0: self.x_coord = scale
            elif axis == 1: self.y_coord = scale

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

def test_ScaledArray(do_plot=0,vectorized_mode=1):

    print("Using array: ",numpy.arange(15.0, 18.8, 0.2))
    #
    # ScaledArray.initialize + set_scale_from_steps
    #
    print("\nTesting ScaledArray.initialize + set_scale_from_steps...")
    scaled_array = ScaledArray.initialize(numpy.arange(15.0, 18.8, 0.2))
    scaled_array.set_scale_from_steps(15.0, 0.2)

    print("interpolated at 18.80 is : ", scaled_array.interpolate_value(18.80))
    print("interpolated at 16.22 is : ", scaled_array.interpolate_value(16.22))
    print("interpolated at 22.35 is : ", scaled_array.interpolate_value(22.35))



    #
    # ScaledArray.initialize + set_scale_from_range ; interpolate vectorized
    #
    print("\nTesting ScaledArray.initialize + set_scale_from_range ; interpolate vectorized...")
    scaled_array = ScaledArray.initialize(numpy.arange(15.0, 18.8, 0.2))
    scaled_array.set_scale_from_range(15.0, 18.8)
    print("interpolated at 18.80 is : ", scaled_array.interpolate_value(18.80))
    print("interpolated at 16.22 is : ", scaled_array.interpolate_value(16.22))
    print("interpolated at 22.35 is : ", scaled_array.interpolate_value(22.35))

    # v_interpolate = numpy.vectorize(scaled_array.interpolate_value)
    # values = v_interpolate([10, 17.3, 16.5, 20, 30])
    # x = [10, 17.3, 16.5, 20, 30]
    # for i in range(len(x)):
    #     print("   interpolated at x=%g is: %g"%(x[i],values[i]))


    #
    # ScaledArray.initialize_from_steps
    #
    print("\nTesting ScaledArray.initialize_from_steps...")
    scaled_array = ScaledArray.initialize_from_steps(numpy.arange(15.0, 18.8, 0.2), 15.0, 0.2)

    print("interpolated at 18.80 is : ", scaled_array.interpolate_value(18.80))
    print("interpolated at 16.22 is : ", scaled_array.interpolate_value(16.22))
    print("interpolated at 22.35 is : ", scaled_array.interpolate_value(22.35))

    #
    # ScaledArray.initialize_from_steps
    #
    print("\nTesting ScaledArray.initialize_from_range...")
    scaled_array = ScaledArray.initialize_from_range(numpy.arange(15.0, 18.8, 0.2),15.0, 18.8)

    #
    # interpolator
    #
    x = numpy.arange(-5.0, 18.8, 3)
    y = x**2
    scaled_array = ScaledArray.initialize_from_range(y,x[0],x[-1])

    x1 = numpy.arange(0, 20.0, 0.1)
    y1 = numpy.zeros_like(x1)



    if vectorized_mode == 0:
        for i in range(x1.size):
            y1[i] = scaled_array.interpolate_value(x1[i])
            print("    interpolated at x=%g is %g (exact=%g): "%(x1[i],y1[i],x1[i]**2))
    else:
        v_interpolate = numpy.vectorize(scaled_array.interpolate_value)
        y1 = v_interpolate(x1)

    for i in range(len(x1)):
        print("   interpolated at x=%g is: %g (exact=%g)"%(x1[i],y1[i],x1[i]**2))


    if do_plot:
        from srxraylib.plot.gol import plot
        plot(x,y,x1,y1,legend=["Data",'Interpolated'],legend_position=[0.4,0.8],
             marker=['','o'],linestyle=['-',''],xrange=[-6,21],yrange=[-5,375])


def test_ScaledMatrix(do_plot=0):

    #
    # Matrix
    #

    x = numpy.arange(100,200,2.0)
    y = numpy.arange(200,250,1.0)
    Z = numpy.array(numpy.meshgrid(y,x))  # Z[0] contains Y, Z[1] contains X

    print("x,Y,Z shape",x.shape,y.shape,Z.shape)

    #
    # scaledMatrix.initialize + set_scale_from_steps
    #
    print("\nTesting ScaledMatrix.initialize + set_scale_from_steps...")
    scaled_matrix = ScaledMatrix.initialize(Z[0])  # Z[0] contains Y
    print("    Matrix shape", scaled_matrix.shape())

    scaled_matrix.set_scale_from_steps(0,x[0],x[1]-x[0])
    scaled_matrix.set_scale_from_steps(1,y[0],y[1]-y[0])

    print("    Matrix X value x=0 : ", scaled_matrix.get_x_value(0.0))
    print("    Matrix Y value x_index=15 is %g (to compare with %g) "%(scaled_matrix.get_x_value(15.0),x[0]+15*(x[1]-x[0])))
    print("    Matrix Y value y_index=15 is %g (to compare with %g) "%(scaled_matrix.get_y_value(15.0),y[0]+15*(y[1]-y[0])))
    print("    Matrix X: ",scaled_matrix.get_x_values()," shape: ",scaled_matrix.get_x_values().shape)
    print("    Matrix Y: ",scaled_matrix.get_y_values()," shape: ",scaled_matrix.get_y_values().shape)
    print("    Matrix Z: ",scaled_matrix.get_z_values()," shape: ",scaled_matrix.get_z_values().shape)


    #
    # scaledMatrix.initialize + set_scale_from_range
    #
    print("\nTesting ScaledMatrix.initialize + set_scale_from_range...")
    scaled_matrix = ScaledMatrix.initialize(Z[0])  # Z[0] contains Y
    print("    Matrix shape", scaled_matrix.shape())

    scaled_matrix.set_scale_from_range(0,x[0],x[-1])
    scaled_matrix.set_scale_from_range(1,y[0],y[-1])

    print("    Matrix X value x=0 : ", scaled_matrix.get_x_value(0.0))
    print("    Matrix Y value x_index=15 is %g (to compare with %g) "%(scaled_matrix.get_x_value(15.0),x[0]+15*(x[1]-x[0])))
    print("    Matrix Y value y_index=15 is %g (to compare with %g) "%(scaled_matrix.get_y_value(15.0),y[0]+15*(y[1]-y[0])))
    print("    Matrix X: ",scaled_matrix.get_x_values()," shape: ",scaled_matrix.get_x_values().shape)
    print("    Matrix Y: ",scaled_matrix.get_y_values()," shape: ",scaled_matrix.get_y_values().shape)
    print("    Matrix Z: ",scaled_matrix.get_z_values()," shape: ",scaled_matrix.get_z_values().shape)




    #
    # scaledMatrix.initialize_from_steps
    #
    print("\nTesting ScaledMatrix.initialize_from_steps...")

    scaled_matrix2 = ScaledMatrix.initialize_from_steps(Z[0],x[0],x[1]-x[0],y[0],y[1]-y[0])

    print("    Matrix 2 shape", scaled_matrix2.shape())
    print("    Matrix 2 value x=0 : ", scaled_matrix2.get_x_value(0.0))
    print("    Matrix 2 value x=15 is %g (to compare with %g) "%(scaled_matrix2.get_x_value(15.0),x[0]+15*(x[1]-x[0])))
    print("    Matrix 2 value y=15 is %g (to compare with %g) "%(scaled_matrix2.get_y_value(15.0),y[0]+15*(y[1]-y[0])))


    #
    # scaledMatrix.initialize_from_range
    #
    print("\nTesting ScaledMatrix.initialize_from_range...")
    #
    scaled_matrix3 = ScaledMatrix.initialize_from_range(Z[0]*0+(3+2j),x[0],x[-1],y[0],y[-1])


    print("Matrix 3 shape", scaled_matrix3.shape())
    print("Matrix 3 value x=0 : ", scaled_matrix3.get_x_value(0.0))
    print("Matrix 3 value x=15 is %g (to compare with %g) "%(scaled_matrix3.get_x_value(15.0),x[0]+15*(x[1]-x[0])))
    print("Matrix 3 value y=15 is %g (to compare with %g) "%(scaled_matrix3.get_y_value(15.0),y[0]+15*(y[1]-y[0])))

    print("Matrix 3 X : ", scaled_matrix3.get_x_values())
    print("Matrix 3 Y : ", scaled_matrix3.get_y_values())
    print("Matrix 3 Z : ", scaled_matrix3.get_z_values())

    #
    # interpolator
    #
    print("\nTesting interpolator...")
    # constant
    scaled_matrix4 = ScaledMatrix.initialize_from_range(Z[0]*Z[1],x[0],x[-1],y[0],y[-1])
    scaled_matrix4.compute_interpolator()

    print("Matrix 4 interpolation x=110,y=210 is ",scaled_matrix4.interpolate_value(110,210))

    #
    # interpolate in the same grid
    #
    s4int = scaled_matrix4.interpolate_value(Z[1],Z[0])
    print("Matrix 4 interpolation same grid (shape)",s4int.shape," before: ",scaled_matrix4.shape())

    #
    # interpolate in a grid with much less points
    #
    xrebin = numpy.linspace(x[0],x[-1],x.size/10)
    yrebin = numpy.linspace(y[0],y[-1],y.size/10)
    Zrebin =numpy.meshgrid( yrebin, xrebin,  )
    s4rebin = scaled_matrix4.interpolate_value(Zrebin[1],Zrebin[0])

    if do_plot == 1:
        from srxraylib.plot.gol import plot_image
        plot_image(scaled_matrix4.get_z_values(),scaled_matrix4.get_x_values(),scaled_matrix4.get_y_values(),
                   title="Original",show=0)
        plot_image(numpy.real(s4int),scaled_matrix4.get_x_values(),scaled_matrix4.get_y_values(),
                   title="Interpolated on same grid",show=0)
        plot_image(numpy.real(s4rebin),xrebin,yrebin,
                   title="Interpolated on a grid with 10 times less points")
if __name__ == "__main__":

    # test_ScaledArray(do_plot=1)
    test_ScaledMatrix(do_plot=1)



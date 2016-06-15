import unittest
import numpy

from srxraylib.util.data_structures import ScaledArray,ScaledMatrix

do_plot = 0

class ScaledArrayTest(unittest.TestCase):

    def test_ScaledArray(self,do_plot=do_plot):



        #
        # ScaledArray.initialize + set_scale_from_steps
        #
        test_array = numpy.arange(15.0, 18.8, 0.2)


        print("\nTesting ScaledArray.initialize + set_scale_from_steps...")


        scaled_array = ScaledArray.initialize(test_array)
        scaled_array.set_scale_from_steps(test_array[0],0.2)

        print("Using array: ",test_array)
        print("Stored array: ",scaled_array.get_values())

        print("Using array: ",test_array)
        print("Stored abscissas: ",scaled_array.get_abscissas())

        numpy.testing.assert_almost_equal(test_array,scaled_array.get_values(),11)
        numpy.testing.assert_almost_equal(test_array,scaled_array.get_abscissas(),11)
        self.assertAlmostEqual(0.2,scaled_array.delta(),11)
        self.assertAlmostEqual(15.0,scaled_array.offset(),11)

        x_in =   (18.80,16.22,22.35)
        x_good = (18.80,16.22,18.8)

        for i,x in enumerate(x_in):
            print("interpolated at %3.2f is: %3.2f (must give %3.2f)"%( x,scaled_array.interpolate_value(x), x_good[i] ))
            self.assertAlmostEqual(scaled_array.interpolate_value(x), x_good[i], 2)


        #
        # ScaledArray.initialize + set_scale_from_range ; interpolate vectorized
        #
        print("\nTesting ScaledArray.initialize + set_scale_from_range ; interpolate vectorized...")
        scaled_array = ScaledArray.initialize(test_array)
        scaled_array.set_scale_from_range(test_array[0],test_array[-1])

        x_in =   (18.80,16.22,22.35)
        x_good = (18.80,16.22,18.8)

        for i,x in enumerate(x_in):
            print("interpolated at %3.2f is: %3.2f (must give %3.2f)"%( x,scaled_array.interpolate_value(x), x_good[i] ))
            self.assertAlmostEqual(scaled_array.interpolate_value(x), x_good[i], 2)


        #
        # ScaledArray.initialize_from_steps
        #
        print("\nTesting ScaledArray.initialize_from_steps...")
        scaled_array = ScaledArray.initialize_from_steps(test_array, test_array[0], 0.2)

        x_in =   (18.80,16.22,22.35)
        x_good = (18.80,16.22,18.8)

        for i,x in enumerate(x_in):
            print("interpolated at %3.2f is: %3.2f (must give %3.2f)"%( x,scaled_array.interpolate_value(x), x_good[i] ))
            self.assertAlmostEqual(scaled_array.interpolate_value(x), x_good[i], 2)

        #
        # ScaledArray.initialize_from_steps
        #
        print("\nTesting ScaledArray.initialize_from_range...")
        scaled_array = ScaledArray.initialize_from_range(test_array,test_array[0], test_array[-1])

        x_in =   (18.80,16.22,22.35)
        x_good = (18.80,16.22,18.8)

        for i,x in enumerate(x_in):
            print("interpolated at %3.2f is: %3.2f (must give %3.2f)"%( x,scaled_array.interpolate_value(x), x_good[i] ))
            self.assertAlmostEqual(scaled_array.interpolate_value(x), x_good[i], 2)


        #
        # interpolator
        #
        print("\nTesting interpolator...")
        x = numpy.arange(-5.0, 18.8, 3)
        y = x**2

        scaled_array = ScaledArray.initialize_from_range(y,x[0],x[-1])

        print("for interpolation; x=",x)
        print("for interpolation; offset, delta:=",scaled_array.offset(),scaled_array.delta())
        print("for interpolation; abscissas:=",scaled_array.get_abscissas())

        x1 = numpy.concatenate( ( numpy.arange(-6, -4, 0.1) , [0], numpy.arange(11, 20.0, 0.1) ) )

        y1 = scaled_array.interpolate_values(x1)

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(x,y,x1,y1,legend=["Data",'Interpolated'],legend_position=[0.4,0.8],
                 marker=['','o'],linestyle=['-',''],xrange=[-6,21],yrange=[-5,375])

        for i in range(len(x1)):
            y2 = x1[i]**2

            if x1[i] <= x[0]:
                y2 = y[0]

            if x1[i] >= x[-1]:
                y2 = y[-1]

            print("   interpolated at x=%g is: %g (expected: %g)"%(x1[i],y1[i],y2))
            self.assertAlmostEqual(1e-3*y1[i], 1e-3*y2, 2)


        # interpolate on same grid
        print("\nTesting interpolation on the same grid...")



        y1 = scaled_array.interpolate_values(scaled_array.get_abscissas())
        if do_plot:
            from srxraylib.plot.gol import plot
            plot(scaled_array.get_abscissas(),scaled_array.get_values(),
                 scaled_array.get_abscissas(),y1,legend=["Data",'Interpolated on same grid'],legend_position=[0.4,0.8],
                 marker=['','o'],linestyle=['-',''])

        numpy.testing.assert_almost_equal(scaled_array.get_values(),y1,5)


class ScaledMatrixTest(unittest.TestCase):


    def test_ScaledMatrix(self,do_plot=do_plot):
        #
        # Matrix
        #

        x = numpy.arange(10,20,2.0)
        y = numpy.arange(20,25,1.0)

        xy = numpy.meshgrid(x,y)
        X = xy[0].T
        Y = xy[1].T
        Z = Y

        print("x,y,X,Y,Z shapes: ",x.shape,y.shape,X.shape,Y.shape,Z.shape)
        print("x,y,X,Y,Z shapes: ",x.shape,y.shape,X.shape,Y.shape,Z.shape)

        #
        # scaledMatrix.initialize + set_scale_from_steps
        #
        print("\nTesting ScaledMatrix.initialize + set_scale_from_steps...")
        scaled_matrix = ScaledMatrix.initialize(Z)
        print("    Matrix shape", scaled_matrix.shape())

        scaled_matrix.set_scale_from_steps(0,x[0],numpy.abs(x[1]-x[0]))
        scaled_matrix.set_scale_from_steps(1,y[0],numpy.abs(y[1]-y[0]))

        print("Original x: ",x)
        print("Stored x:   ",scaled_matrix.get_x_values())
        print("Original y: ",y)
        print("Stored y:    ",scaled_matrix.get_y_values())
        numpy.testing.assert_equal(x,scaled_matrix.get_x_values())
        numpy.testing.assert_equal(y,scaled_matrix.get_y_values())

        print("    Matrix X value x=0 : ", scaled_matrix.get_x_value(0.0))
        print("    Matrix Y value x_index=3 is %g (to compare with %g) "%(scaled_matrix.get_x_value(2.0),x[0]+2*(x[1]-x[0])))
        print("    Matrix Y value y_index=3 is %g (to compare with %g) "%(scaled_matrix.get_y_value(2.0),y[0]+2*(y[1]-y[0])))
        self.assertAlmostEqual(scaled_matrix.get_x_value(2.0),x[0]+2*(x[1]-x[0]),10)
        self.assertAlmostEqual(scaled_matrix.get_y_value(2.0),y[0]+2*(y[1]-y[0]),10)


        #
        # scaledMatrix.initialize + set_scale_from_range
        #

        x = numpy.linspace(-10,10,10)
        y = numpy.linspace(-25,25,20)

        xy = numpy.meshgrid(x,y)
        X = xy[0].T
        Y = xy[1].T
        Z = Y

        print("\nTesting ScaledMatrix.initialize + set_scale_from_range...")
        scaled_matrix = ScaledMatrix.initialize(Z)  # Z[0] contains Y
        print("    Matrix shape", scaled_matrix.shape())

        scaled_matrix.set_scale_from_range(0,x[0],x[-1])
        scaled_matrix.set_scale_from_range(1,y[0],y[-1])

        print("Original x: ",x)
        print("Stored x:   ",scaled_matrix.get_x_values())
        print("Original y: ",y)
        print("Stored y:    ",scaled_matrix.get_y_values())

        numpy.testing.assert_almost_equal(x,scaled_matrix.get_x_values(),11)
        numpy.testing.assert_almost_equal(y,scaled_matrix.get_y_values(),11)

        print("    Matrix X value x=0 : ", scaled_matrix.get_x_value(0.0))
        print("    Matrix Y value x_index=5 is %g (to compare with %g) "%(scaled_matrix.get_x_value(5.0),x[0]+5*(x[1]-x[0])))
        print("    Matrix Y value y_index=5 is %g (to compare with %g) "%(scaled_matrix.get_y_value(5.0),y[0]+5*(y[1]-y[0])))
        self.assertAlmostEqual(scaled_matrix.get_x_value(5.0),x[0]+5*(x[1]-x[0]),10)
        self.assertAlmostEqual(scaled_matrix.get_y_value(5.0),y[0]+5*(y[1]-y[0]),10)



        #
        # scaledMatrix.initialize_from_steps
        #

        x = numpy.arange(10,20,0.2)
        y = numpy.arange(20,25,0.1)

        xy = numpy.meshgrid(x,y)
        X = xy[0].T
        Y = xy[1].T
        Z = Y

        print("\nTesting ScaledMatrix.initialize_from_steps...")

        scaled_matrix2 = ScaledMatrix.initialize_from_steps(Z,x[0],numpy.abs(x[1]-x[0]),y[0],numpy.abs(y[1]-y[0]))

        print("    Matrix 2 shape", scaled_matrix2.shape())
        print("    Matrix 2 value x=0 : ", scaled_matrix2.get_x_value(0.0))

        print("    Matrix 2 value x=5 is %g (to compare with %g) "%(scaled_matrix2.get_x_value(5.0),x[0]+5*(x[1]-x[0])))
        self.assertAlmostEqual(scaled_matrix2.get_x_value(5.0),x[0]+5*(x[1]-x[0]))
        print("    Matrix 2 value y=5 is %g (to compare with %g) "%(scaled_matrix2.get_y_value(5.0),y[0]+5*(y[1]-y[0])))
        self.assertAlmostEqual(scaled_matrix2.get_y_value(5.0),y[0]+5*(y[1]-y[0]))


        #
        # scaledMatrix.initialize_from_range
        #
        print("\nTesting ScaledMatrix.initialize_from_range...")

        x = numpy.linspace(-10,10,20)
        y = numpy.linspace(-25,25,30)

        xy = numpy.meshgrid(x,y)
        X = xy[0].T
        Y = xy[1].T
        Z = X*0+(3+2j)

        #
        scaled_matrix3 = ScaledMatrix.initialize_from_range(Z,x[0],x[-1],y[0],y[-1])

        print("Matrix 3 shape", scaled_matrix3.shape())
        print("Matrix 3 value x=0 : ", scaled_matrix3.get_x_value(0.0))

        print("Matrix 3 value x=15 is %g (to compare with %g) "%(scaled_matrix3.get_x_value(15.0),x[0]+15*(x[1]-x[0])))
        self.assertAlmostEqual(scaled_matrix3.get_x_value(15.0),x[0]+15*(x[1]-x[0]))
        print("Matrix 3 value y=15 is %g (to compare with %g) "%(scaled_matrix3.get_y_value(15.0),y[0]+15*(y[1]-y[0])))
        self.assertAlmostEqual(scaled_matrix3.get_y_value(15.0),y[0]+15*(y[1]-y[0]))

        # print("Matrix 3 X : ", scaled_matrix3.get_x_values())
        # print("Matrix 3 Y : ", scaled_matrix3.get_y_values())
        # print("Matrix 3 Z : ", scaled_matrix3.get_z_values())

        #
        # interpolator
        #
        print("\nTesting interpolator on the same grid...")
        # constant
        Z = X * Y
        scaled_matrix4 = ScaledMatrix.initialize_from_range(Z,x[0],x[-1],y[0],y[-1])
        scaled_matrix4.compute_interpolator()

        print("Matrix 4 interpolation x=110,y=210 is ",scaled_matrix4.interpolate_value(110,210))

        # interpolate in the same grid
        s4int = scaled_matrix4.interpolate_value(X,Y)
        print("Matrix 4 interpolation same grid (shape)",s4int.shape," before: ",scaled_matrix4.shape())
        numpy.testing.assert_almost_equal(scaled_matrix4.get_z_values(),s4int,3)


        print("\nTesting interpolator on a grid with less points...")
        # interpolate in a grid with much less points
        xrebin = numpy.linspace(x[0],x[-1],x.size/10)
        yrebin = numpy.linspace(y[0],y[-1],y.size/10)
        xyrebin =numpy.meshgrid( xrebin, yrebin)
        Xrebin = xyrebin[0].T
        Yrebin = xyrebin[1].T

        s4rebin = scaled_matrix4.interpolate_value(Xrebin,Yrebin)

        if do_plot == 1:
            from srxraylib.plot.gol import plot_image
            plot_image(scaled_matrix4.get_z_values(),scaled_matrix4.get_x_values(),scaled_matrix4.get_y_values(),
                       title="Original",show=0)
            plot_image(numpy.real(s4int),scaled_matrix4.get_x_values(),scaled_matrix4.get_y_values(),
                       title="Interpolated on same grid",show=0)
            plot_image(numpy.real(s4rebin),xrebin,yrebin,
                       title="Interpolated on a grid with 10 times less points")

        for xi in xrebin:
            for yi in yrebin:
                yint = scaled_matrix4.interpolate_value(xi,yi)
                print("    Interpolated at (%g,%g) is %g+%gi (expected %g)"%(xi,yi,yint.real,yint.imag,xi*yi))
                self.assertAlmostEqual(scaled_matrix4.interpolate_value(xi,yi),xi*yi)

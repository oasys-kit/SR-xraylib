
__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"
__date__ = "30/08/2018"


import numpy
from srxraylib.util.inverse_method_sampler import Sampler1D, Sampler2D, Sampler3D
from scipy import interpolate
from srxraylib.plot.gol import plot, plot_image, plot_scatter, plot, set_qt

# used in test2d_bis()
from PIL import Image
import requests
from io import BytesIO

import unittest

RUN_1D = True
RUN_2D = True
RUN_3D = True


do_plots = False
set_qt()

def Std_zero_mean(array): # std for zero mean!!!
    return numpy.sqrt( (array**2).sum() / (array.size-1) )

def histogram_path(histogram,bin_edges):
    bin_center = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) * 0.5
    bin_left = bin_edges[:-1]
    bin_right = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])

    tmp_b = []
    tmp_h = []

    for s, t, v in zip(bin_left, bin_right, histogram):
        tmp_b.append(s)
        tmp_h.append(v)
        tmp_b.append(t)
        tmp_h.append(v)

    histogram_path = numpy.array(tmp_h)
    bin_path = numpy.array(tmp_b)
    return histogram_path,bin_path


class TestSamplers(unittest.TestCase):

    @unittest.skipUnless(RUN_1D,"Running 1D Test")
    def test_1d_constant(self):

        print("\n#\n# running test_1d() \n#\n")

        x = numpy.linspace(0.0,10,100)
        y = numpy.ones_like(x)
        y[50:55] = 0.5


        s1 = Sampler1D(y,x)

        if do_plots:
            plot(s1.abscissas(),s1.pdf(),title="pdf",marker='+')
            plot(s1.abscissas(),s1.pdf()/s1.pdf().max(),
                 s1.abscissas(),s1.cdf(),legend=["pdf normalized","cdf"],marker=['+','+'])


        #
        # defingn random points
        #
        cdf_rand_array = numpy.random.random(100000)
        sampled_points, h_center, bin_edges = s1.get_sampled_and_histogram(cdf_rand_array,range=[x.min(),x.max()])

        h_path, bin_path = histogram_path(h_center,bin_edges)
        model_x = s1.abscissas()
        model_y = s1.pdf()/s1.pdf().max()

        print("Min: %d (pdf: %d), Max: %f (pdf: %f) "%(sampled_points.min(),y.min(),sampled_points.max(),y.max()))

        if do_plots:
            # plot(numpy.arange(cdf_rand_array.size),sampled_points,title="sampled points")
            plot(bin_path,h_path/h_path.max(),model_x,model_y,title="histogram",legend=["histo","data"],marker=['+','+'])


        assert( sampled_points.min() < x[1])
        assert (sampled_points.max() > x[-2])


    @unittest.skipUnless(RUN_1D,"Running 1D Test")
    def test_1d(self):

        print("\n#\n# running test_1d() \n#\n")
        npoints = 51
        x = numpy.linspace(-10,10,npoints)
        y = numpy.zeros_like(x)

        y[0:21] = 100.0
        y[21:31] = 4.0
        y[31:41] = 5.0
        y[41:51] = 10.0


        s1 = Sampler1D(y,x,cdf_interpolation_factor=2)


        if do_plots:
            plot(s1.abscissas(),s1.pdf()/s1.pdf().max(),
                 s1.cdf_abscissas(),s1.cdf(),legend=["pdf normalized","cdf (cdf_interpolation_factor=2)"],marker=['+','+'])

        #
        # defingn random points
        #
        cdf_rand_array = numpy.random.random(100000)
        sampled_points,h_center,bin_center = s1.get_sampled_and_histogram(cdf_rand_array,bins=(npoints-1))

        h_path, bin_path = histogram_path(h_center,bin_center)
        model_x = s1.abscissas()
        model_y = s1.pdf()/s1.pdf().max()

        if do_plots:

            plot( 0.5*(bin_center[:-1]+bin_center[1:]),h_center/h_center.max(),
                  bin_path,h_path/h_path.max(),
                  model_x,model_y,
                  title="histogram",legend=["histo (points)","histo (bins)","data"],marker=['+','+','+'])

        # interpolate for comparing model with resulting histogram
        fn = interpolate.interp1d(model_x, model_y)
        sampled_y_interpolated = fn(bin_center)

        diff = sum( (1./sampled_y_interpolated.size) * (model_y-sampled_y_interpolated)**2 )
        assert(diff < 1e-2)


    @unittest.skipUnless(RUN_2D,"Running 2D Test")
    def test2d_radial(self):
        import h5py
        from srxraylib.plot.gol import plot_image
        r = numpy.linspace(0,5,100)
        theta = numpy.linspace(0,2*numpy.pi,100)

        R = numpy.outer(r,numpy.ones_like(theta))
        T = numpy.outer(numpy.ones_like(r),theta)
        Z = R

        if do_plots:
            plot_image(Z, r, theta)

        s2d = Sampler2D(Z, r , theta)
        sampled_points_r, sampled_points_theta = s2d.get_n_sampled_points(100000)

        X = sampled_points_r * numpy.cos(sampled_points_theta)
        Y = sampled_points_r * numpy.sin(sampled_points_theta)

        if do_plots:
            plot_scatter(X,Y)
        # assert(False)

    @unittest.skipUnless(RUN_2D, "Running 2D Test")
    def test2d_shadow_undulator(self):
        import h5py
        from srxraylib.plot.gol import plot_image
        import Shadow
        f = h5py.File("/home/manuel/Downloads/manolone.hdf5","r")
        X = f["surface_file/X"][:]
        Y = f["surface_file/Y"][:]
        Z = f["surface_file/Z"][:].T
        f.close()
        # plot_image(Z,X,Y)
        X = numpy.linspace(0,10,X.size)
        Y = numpy.linspace(0, 10, Y.size)

        s2d = Sampler2D(Z, X , Y)
        sampled_points_x, sampled_points_y = s2d.get_n_sampled_points(100000)
        S = Shadow.Beam(sampled_points_x.size)
        S.rays[:,0] = sampled_points_x
        S.rays[:,2] = sampled_points_y
        S.rays[:,16] = 1
        S.rays[:, 9] = 1
        if do_plots:
            Shadow.ShadowTools.plotxy(S,1,3,nbins_h=101,nbins_v=101)
            # S.write("/Users/srio/Oasys/tmp.dat")
            plot_scatter(sampled_points_x,sampled_points_y)

    @unittest.skipUnless(RUN_2D,"Running 2D Test")
    def test_2d(self):

        print("\n#\n# running test_2d() \n#\n")
        #
        response = requests.get("https://cdn104.picsart.com/201671193005202.jpg?r1024x1024")
        # response = requests.get("https://www.lbl.gov/wp-content/uploads/2013/06/Lawrence-tb.jpg")

        img = Image.open(BytesIO(response.content))
        img = numpy.array(img).sum(2) * 1.0
        img = numpy.rot90(img,axes=(1,0))
        image_data = img.max() - img
        if do_plots:
            plot_image(image_data,cmap='binary')

        x0 = numpy.arange(image_data.shape[0])
        x1 = numpy.arange(image_data.shape[1])

        print(image_data.shape)

        s2d = Sampler2D(image_data,x0,x1)

        # plot_image(s2d.pdf(),cmap='binary',title="pdf")

        cdf2,cdf1 = s2d.cdf()
        print("cdf2.shape,cdf1.shape,s2d._pdf_x0.shape,s2d._pdf_x1.shape: ",cdf2.shape,cdf1.shape,s2d._pdf_x0.shape,s2d._pdf_x1.shape)
        # plot_image(cdf2,cmap='binary',title="cdf")
        # plot(s2d.abscissas()[0],s2d.cdf()[0][:,-1])
        # plot(s2d.abscissas()[0],cdf1)

        x0s,x1s = s2d.get_n_sampled_points(100000)
        if do_plots:
            plot_scatter(x0s,x1s)

        print("x0s.mean(),x0s.std(),x1s.mean(),x1s.std(): ",x0s.mean(),x0s.std(),x1s.mean(),x1s.std())
        assert ((numpy.abs( x0s.mean() ) - 731.37) < 10.0)
        assert ((numpy.abs( x0s.std() )  - 458.55) < 10.0)
        assert ((numpy.abs( x1s.mean() ) - 498.05) < 10.0)
        assert ((numpy.abs( x1s.std() )  - 301.05) < 10.0)

    @unittest.skipUnless(RUN_3D,"Running 3D Test")
    def test3d(self):


        print("\n#\n# running test_3d() \n#\n")


        response = requests.get("https://cdn104.picsart.com/201671193005202.jpg?r1024x1024")

        img = Image.open(BytesIO(response.content))
        img = numpy.array(img) * 1.0
        img = numpy.rot90(img,axes=(1,0))
        image_data = img.max() - img

        if do_plots:
            plot_image(image_data[:,:,0],cmap='binary',title="channel0",show=1)
        # plot_image(image_data[:,:,1],cmap='binary',title="channel1",show=0)
        # plot_image(image_data[:,:,2],cmap='binary',title="channel2")

        s2d = Sampler3D(image_data)


        cdf3, cdf2, cdf1 = s2d.cdf()

        # plot_image(cdf2)
        #
        # plot_image(s2d.pdf(),cmap='binary',title="pdf")
        #
        # cdf2,cdf1 = s2d.cdf()
        # plot_image(cdf2,cmap='binary',title="cdf")
        # plot(s2d.abscissas()[0],s2d.cdf()[0][:,-1])
        # plot(s2d.abscissas()[0],cdf1)
        #
        x0s,x1s,x2s = s2d.get_n_sampled_points(100000)
        if do_plots:
            plot_scatter(x0s,x1s)

        print("x0s.mean(),x0s.std(),x1s.mean(),x1s.std(): ",x0s.mean(),x0s.std(),x1s.mean(),x1s.std())
        assert ((numpy.abs( x0s.mean() ) - 730.09)  < 10.0)
        assert ((numpy.abs( x0s.std() )  - 458.17)  < 10.0)
        assert ((numpy.abs( x1s.mean() ) - 498.805) < 10.0)
        assert ((numpy.abs( x1s.std() )  - 301.21)  < 10.0)

    @unittest.skipUnless(RUN_1D,"Running 1D Test")
    def test_radial_1D_gaussian_distribution(self):


        print("\n#\n# running test_radial_1D_gaussian_distribution() \n#\n")


        R = 50.0e-6
        sigma = 5e-6
        NRAYS = 10000


        x = numpy.linspace(0,R,300)
        y = numpy.exp(- x*x/2/sigma/sigma) # numpy.ones_like(x) * R

        s = Sampler1D(y*x,x)

        sampled_point,  hy, hx = s.get_n_sampled_points_and_histogram(NRAYS,bins=101)

        hy_path, hx_path = histogram_path(hy, hx)

        angle = numpy.random.random(NRAYS) * 2 * numpy.pi

        if do_plots:
            plot(x,y,title="Gaussian Radial Distribution sigma:%f"%sigma)
            plot(x,y*x,title="pdf")
            plot(hx_path,hy_path,title="histogram of sampled r")

        X = sampled_point / numpy.sqrt(2) * numpy.sin(angle)
        Y = sampled_point / numpy.sqrt(2) * numpy.cos(angle)



        if do_plots:
            plot_scatter(X,Y)

        print("sigma, stdev R,  stdev X, srdev Y: ",sigma,Std_zero_mean(numpy.sqrt(X*X+Y*Y)),Std_zero_mean(X),Std_zero_mean(Y))
        print("stdev sampled R: ",Std_zero_mean(sampled_point))
        print("error (percent) : ",100/sigma*numpy.abs(sigma - (Std_zero_mean(numpy.sqrt(X*X+Y*Y)))) )

        assert ( numpy.abs(sigma - (Std_zero_mean(numpy.sqrt(X*X+Y*Y)))) < 0.05 * sigma) # 5% difference



    @unittest.skipUnless(RUN_2D,"Running 2D Test")
    def test_cartesian_2D_gaussian_distribution(self):


        print("\n#\n# running test_cartesian_2D_gaussian_distribution() \n#\n")


        R = 50.0e-6
        sigma = 5e-6

        x = numpy.linspace(-R, R, 100)
        y = numpy.linspace(-R, R, 100)

        X = numpy.outer(x,numpy.ones_like(y))
        Y = numpy.outer(numpy.ones_like(x),y)
        G = numpy.exp(- (X**2 + Y**2)/2/sigma/sigma) * R

        s2d = Sampler2D(G,x,y)

        if do_plots:
            # plot_image(G,r,theta,aspect='auto',xtitle="r",ytitle="theta")
            cdf2,cdf1=s2d.cdf()
            print(cdf2.shape,cdf1.shape)
            plot_image(cdf2,x,y,aspect='auto')

        sampled_points_x, sampled_points_y = s2d.get_n_sampled_points(100000)

        if do_plots:
            plot_scatter(sampled_points_x,sampled_points_y)

        assert ( (numpy.abs(sampled_points_x.std() - sigma) < sigma))
        assert ( (numpy.abs(sampled_points_y.std() - sigma) < sigma))


    @unittest.skipUnless(RUN_2D,"Running 2D Test")
    def test_radial_2D_gaussian_distribution(self):


        print("\n#\n# running test_radial_2D_gaussian_distribution() \n#\n")


        R = 50.0e-6
        sigma = 5e-6
        NRAYS = 10000

        Nr = 100
        Nt = 60
        r = numpy.linspace(0,R,Nr)
        theta = numpy.linspace(0,2*numpy.pi,Nt)

        R = numpy.outer(r,numpy.ones_like(theta))
        T = numpy.outer(numpy.ones_like(r),theta)
        G = numpy.exp(- R*R/2/sigma/sigma) * R

        s2d = Sampler2D(G,r,theta)

        if do_plots:
            cdf2,cdf1=s2d.cdf()
            plot_image(G, r, theta, aspect='auto',title="pdf",show=0)
            plot_image(cdf2,r,theta,aspect='auto',title='cdf2',show=0)
            plot(r,cdf1,title='cdf1')



        sampled_points_r, sampled_points_theta = s2d.get_n_sampled_points(100000)


        print(">>>> NaN sample_points_r ", numpy.isnan(sampled_points_r).any())
        print(">>>> NaN sample_points_t ", numpy.isnan(sampled_points_theta).any())


        #
        X = sampled_points_r / numpy.sqrt(2) * numpy.sin(sampled_points_theta)
        Y = sampled_points_r / numpy.sqrt(2) * numpy.cos(sampled_points_theta)


        if do_plots:
            plot_scatter(X,Y)
            plot_scatter(sampled_points_r,sampled_points_theta)


        assert ( (numpy.abs(X.std() - sigma) < sigma))


    @unittest.skipUnless(RUN_1D,"Running 1D Test")
    def test_radial_1D_flat_distribution(self):


        print("\n#\n# running test_radial_1D_flat_distribution() \n#\n")

        R = 50.0e-6
        NRAYS = 10000


        x = numpy.linspace(0,R,300)
        y = numpy.ones_like(x)

        s = Sampler1D(y*x,x)

        sampled_point, hy, hx = s.get_n_sampled_points_and_histogram(NRAYS,bins=101)


        angle = numpy.random.random(NRAYS) * 2 * numpy.pi

        if do_plots:
            plot(x,y,title="Constant Radial Distribution")
            plot(x,y*x,title="pdf")
            plot(0.5*(hx[:-1]+hx[1:]),hy,title="histogram of sampled r")

        X = sampled_point * numpy.sin(angle)
        Y = sampled_point * numpy.cos(angle)


        if do_plots:
            plot_scatter(X,Y)

        RR = numpy.sqrt(X*X+Y*Y)
        assert ( numpy.abs(RR.max()) - R < 0.05 * R) # 5% difference

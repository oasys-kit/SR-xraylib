
__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"
__date__ = "30/08/2018"


import numpy
from srxraylib.util.inverse_method_sampler import Sampler1D, Sampler2D, Sampler3D
from scipy import interpolate
from srxraylib.plot.gol import plot, plot_image, plot_scatter, plot

# used in test2d_bis()
from PIL import Image
import requests
from io import BytesIO

import unittest


do_plots = False

def Std_zero_mean(array): # std for zero mean!!!
    return numpy.sqrt( (array**2).sum() / (array.size-1) )

class TestSamplers(unittest.TestCase):
    def test_1d(self):

        print("\n#\n# running test_1d() \n#\n")

        x0=0.0
        sigma=2.0
        x = numpy.linspace(-10,10,51)
        y = numpy.exp(- (x-x0)**2 / 2 / sigma**2)

        y[0:21] = 100.0
        y[21:31] = 4.0
        y[31:41] = 5.0
        y[41:51] = 10.0


        s1 = Sampler1D(y,x)

        if do_plots:
            plot(s1.abscissas(),s1.pdf(),title="pdf")
            plot(s1.abscissas(),s1.cdf(),title="cdf")


        #
        # defingn random points
        #
        cdf_rand_array = numpy.random.random(100000)
        sampled_points,h,hx = s1.get_sampled_and_histogram(cdf_rand_array)

        model_x = s1.abscissas()
        model_y = s1.pdf()/s1.pdf().max()
        sampled_x = hx
        sampled_y = h/h.max()

        if do_plots:
            plot(numpy.arange(cdf_rand_array.size),sampled_points,title="sampled points")
            plot(sampled_x,sampled_y,model_x,model_y,title="histogram",legend=["histo","data"])

        # interpolate for comparing model with resulting histogram
        fn = interpolate.interp1d(model_x, model_y)
        sampled_y_interpolated = fn(sampled_x)

        diff = sum( (1./sampled_y_interpolated.size) * (model_y-sampled_y_interpolated)**2 )
        assert(diff < 1e-2)

    def test_1d_bis(self):

        print("\n#\n# running test_1d_bis() \n#\n")

        x0=0.0
        sigma=2.0
        x = numpy.linspace(-10,10,51)
        y = numpy.exp(- (x-x0)**2 / 2 / sigma**2)

        y[0:21] = 100.0
        y[21:31] = 4.0
        y[31:41] = 5.0
        y[41:51] = 10.0


        s1 = Sampler1D(y,x)

        if do_plots:
            plot(s1.abscissas(),s1.pdf(),title="pdf")
            plot(s1.abscissas(),s1.cdf(),title="cdf")


        #
        # defining N
        #
        sampled_points,h,hx = s1.get_n_sampled_points_and_histogram(120000)

        model_x = s1.abscissas()
        model_y = s1.pdf()/s1.pdf().max()
        sampled_x = hx
        sampled_y = h/h.max()


        if do_plots:
            plot(numpy.arange(120000),sampled_points,title="120000 sampled points")
            plot(sampled_x,sampled_y,model_x,model_y,title="histogram",legend=["histo","data"])

        # interpolate for comparing model with resulting histogram
        fn = interpolate.interp1d(model_x, model_y)
        sampled_y_interpolated = fn(sampled_x)

        diff = sum( (1./sampled_y_interpolated.size) * (model_y-sampled_y_interpolated)**2 )
        assert(diff < 1e-2)



    def test_2d(self):

        print("\n#\n# running test_2d() \n#\n")
        #
        response = requests.get("https://cdn104.picsart.com/201671193005202.jpg?r1024x1024")

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


        print(image_data.shape)

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
        print(x2s)

        print("x0s.mean(),x0s.std(),x1s.mean(),x1s.std(): ",x0s.mean(),x0s.std(),x1s.mean(),x1s.std())
        assert ((numpy.abs( x0s.mean() ) - 730.09)  < 10.0)
        assert ((numpy.abs( x0s.std() )  - 458.17)  < 10.0)
        assert ((numpy.abs( x1s.mean() ) - 498.805) < 10.0)
        assert ((numpy.abs( x1s.std() )  - 301.21)  < 10.0)

    def test_radial_1D_gaussian_distribution(self):


        print("\n#\n# running test_radial_1D_gaussian_distribution() \n#\n")


        R = 50.0e-6
        sigma = 5e-6
        NRAYS = 10000


        x = numpy.linspace(0,R,300)
        y = numpy.exp(- x*x/2/sigma/sigma) # numpy.ones_like(x) * R

        s = Sampler1D(y*x,x)

        sampled_point, hy, hx = s.get_n_sampled_points_and_histogram(NRAYS,bins=101)


        angle = numpy.random.random(NRAYS) * 2 * numpy.pi

        if do_plots:
            plot(x,y,title="Gaussian Radial Distribution sigma:%f"%sigma)
            plot(x,y*x,title="pdf")
            plot(hx,hy,title="histogram of sampled r")

        X = sampled_point / numpy.sqrt(2) * numpy.sin(angle)
        Y = sampled_point / numpy.sqrt(2) * numpy.cos(angle)



        if do_plots:
            plot_scatter(X,Y)

        print("sigma, stdev R,  stdev X, srdev Y: ",sigma,Std_zero_mean(numpy.sqrt(X*X+Y*Y)),Std_zero_mean(X),Std_zero_mean(Y))
        print("stdev sampled R: ",Std_zero_mean(sampled_point))
        print("error (percent) : ",100/sigma*numpy.abs(sigma - (Std_zero_mean(numpy.sqrt(X*X+Y*Y)))) )

        assert ( numpy.abs(sigma - (Std_zero_mean(numpy.sqrt(X*X+Y*Y)))) < 0.05 * sigma) # 5% difference



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
            plot(hx,hy,title="histogram of sampled r")

        X = sampled_point * numpy.sin(angle)
        Y = sampled_point * numpy.cos(angle)


        if do_plots:
            plot_scatter(X,Y)

        RR = numpy.sqrt(X*X+Y*Y)
        assert ( numpy.abs(RR.max()) - R < 0.05 * R) # 5% difference
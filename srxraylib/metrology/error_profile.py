import numpy

def enoise(x):
    tmp = numpy.random.rand()
    return x*tmp

FIGURE_ERROR = 0
SLOPE_ERROR = 1

GAUSSIAN = 0
FRACTAL = 0

def create_random_rough_surface_1D(n_surface_points=100, mirror_length=200.0, rms_height=3e-9, correlation_length=0.03, profile_type=GAUSSIAN):
    # function [f,x] = rsgeng1D(N,rL,h,cl)
    # %
    # % [f,x] = rsgeng1D(N,rL,h,cl)
    # %
    # % generates a 1-dimensional random rough surface f(x) with N surface points.
    # % The surface has a Gaussian height distribution function and a Gaussian
    # % autocovariance function, where rL is the length of the surface, h is the
    # % RMS height and cl is the correlation length.
    # %
    # % Input:    N   - number of surface points
    # %           rL  - length of surface
    # %           h   - rms height
    # %           cl  - correlation length
    # %
    # % Output:   f   - surface heights
    # %           x   - surface points
    # %
    # % Last updated: 2010-07-26 (David Bergström).
    # %
    #

    if profile_type==GAUSSIAN:
        # format long;
        #
        # x = linspace(-rL/2,rL/2,N);
        x_coords = numpy.linspace(-mirror_length / 2, mirror_length / 2, n_surface_points)

        #
        # Z = h.*randn(1,N); % uncorrelated Gaussian random rough surface distribution
        #                      % with mean 0 and standard deviation h
        #
        uncorrelated_gaussian_random_rough_surface = rms_height * numpy.random.randn(1.0, n_surface_points)
        uncorrelated_gaussian_random_rough_surface.shape = -1

        # % Gaussian filter
        # F = exp(-x.^2/(cl^2/2));
        gaussian_filter = numpy.exp(-x_coords**2 / (correlation_length ** 2 / 2))
        #
        # % correlation of surface using convolution (faltung), inverse
        # % Fourier transform and normalizing prefactors
        # f = sqrt(2/sqrt(pi))*sqrt(rL/N/cl)*ifft(fft(Z).*fft(F));
        y_values = numpy.sqrt(2/numpy.sqrt(numpy.pi))*numpy.sqrt(mirror_length / n_surface_points / correlation_length) * numpy.fft.ifft(numpy.fft.fft(uncorrelated_gaussian_random_rough_surface) * numpy.fft.fft(gaussian_filter))
    elif profile_type==FRACTAL:
        x_coords = numpy.linspace(-0.5*mirror_length,0.5*mirror_length,N)
        xstep = mirror_length/(n_surface_points-1)
        freq = numpy.linspace(1/(1*mirror_length),1/(4*xstep),500)
        ampl = freq**(-0.9)
        phases = numpy.random.rand(freq.size)*2*numpy.pi
        ymirr = numpy.zeros(n_surface_points)
        for i in range(len(freq)):
            ymirr += (ampl[i] *  numpy.sin(2*numpy.pi*freq[i]*x_coords + phases[i]))

        y_values = ymirr / ymirr.std() * rms_height
    else:
        raise Exception("Profile type not recognized")

    return x_coords, y_values


def create_simulated_1D_file_APS(mirror_length=200.0, step=1, random_seed=8787, error_type=FIGURE_ERROR, rms=1e-6):
    """
    :param mirror_length: "Enter mirror length, even"
    :param step: "Step for length "
    :param random_seed: "Random seed between 0 and 1"
    :param error_type: "Figure error (0) or Slope error (1)"
    :param rms: "RMS value of the above"
    :return: x coords, y values
    """
    if(step ==0):
        mirror_length=200.0	#Number of points surface wave
        step=1			      #Spacing surface wave
        random_seed=8787
        error_type=FIGURE_ERROR
        rms=0.1e-6

    numpy.random.seed(seed=random_seed)
    print ("mirrorLength: %f, Step: %f, RandomSeed: %d, SEorFE: %d, RMS: %g" % (mirror_length, step, random_seed, error_type, rms))

    mult1 = 2.1e-10
    mult2 = mult1
    slo1 = -1.5
    slo2 = slo1
    chSlo = 0.001

    npo=int(mirror_length / step + 1)

    error_profile_x = numpy.linspace(-mirror_length / 2, mirror_length / 2, npo)
    error_profile = numpy.zeros(npo)

    freq= 1.0 / mirror_length

    x = numpy.linspace(freq, freq + freq * ((npo-1) * step), npo)
    FouAmp = numpy.zeros(npo)
    FouPha = numpy.zeros(npo)
    FouFre=x

    for i in range(npo):
        if (FouFre[i] < chSlo):
            FouAmp[i]=mult1*FouFre[i]**slo1
        else:
            FouAmp[i]=mult2*FouFre[i]**slo2

    for i in range(npo):
        FouPha[i] = enoise(numpy.pi)
        error_profile += FouAmp[i]*numpy.cos(-numpy.pi*2*error_profile_x*FouFre[i]+FouPha[i])

    if (error_type == SLOPE_ERROR): # :TODO check this passage!!!!!
        SF_DIF = numpy.gradient(error_profile, step)
        V_sdev = SF_DIF.std()
        error_profile *= rms / V_sdev
    elif error_type == FIGURE_ERROR:
        V_sdev = error_profile.std()
        error_profile *= rms / V_sdev

    return error_profile_x, error_profile

def create_simulated_2D_profile_APS(mirror_length=200.0, step_l=1.0, random_seed_l=8787, error_type_l=FIGURE_ERROR, rms_l=1e-6,
                                    mirror_width=20.0, step_w=1.0, random_seed_w=8788, error_type_w=FIGURE_ERROR, rms_w=1e-6):
    """
    :param mirror_length: "Enter mirror length, even"
    :param step_l: "Step folr length "
    :param random_seed_l: "Random seed between 0 and 1"
    :param error_type_l: "Figure error (0) or Slope error (1)"
    :param rms_l: "RMS value of the above"
    :param mirror_width: "Enter mirror Width, even"
    :param step_w: "Step for width"
    :param random_seed_w: "Random seed between 0 and 1"
    :param error_type_w: "Figure error (0) or Slope error (1)"
    :param rms_w: "RMS value of the above"
    :return: x coords, y coords, z values
    """
    numpy.random.seed(seed=random_seed_l)

    if step_l == 0:
        mirror_length=200		#Number of points surface wave
        step_l=1			#Spacing surface wave
        random_seed_l=0.1
        error_type_l=FIGURE_ERROR
        rms_l=0.1e-6
        mirror_width=20		#Number of points surface wave
        step_w=1			#Spacing surface wave
        random_seed_w=0.2
        error_type_w=FIGURE_ERROR
        rms_w=1e-6

    WW_x, WW = create_simulated_1D_file_APS(mirror_width, step_w, random_seed_w, error_type_w, rms_w)
    SF_x, SF = create_simulated_1D_file_APS(mirror_length, step_l, random_seed_l, error_type_l, rms_l)

    npoL = SF.size
    npoW = WW.size

    s = numpy.zeros((npoL, npoW))
    for i in range(npoW):
        s[:,i] = SF + WW[i]

    return WW_x, SF_x, s

def create_2D_profile_from_1D(profile_1D_x, profile_1D_y, mirror_width=20.0, step_w=1.0, random_seed_w=8787, error_type_w=FIGURE_ERROR, rms_w=1e-6):
    numpy.random.seed(seed=random_seed_w)

    if step_w == 0:
        mirror_width=20		#Number of points surface wave
        step_w=1			#Spacing surface wave
        random_seed_w=0.2
        error_type_w=FIGURE_ERROR
        rms_w=1e-6

    WW_x, WW = create_simulated_1D_file_APS(mirror_width, step_w, random_seed_w, error_type_w, rms_w)
    SF_x, SF = profile_1D_x, profile_1D_y

    npoL = SF.size
    npoW = WW.size

    s = numpy.zeros((npoL, npoW))
    for i in range(npoW):
        s[:,i] = SF + WW[i]

    return WW_x, SF_x, s

#########################################################
#
# TESTS
#
#########################################################

def test_1d():
    mirrorLength = 200.0 # mm
    Step = 1.0
    RandomSeed = 898882
    SEorFE = FIGURE_ERROR # 0 = Figure, 1=Slope
    RMS = 1e-7 # mm (0.1 nm)
    wName_x,wName = create_simulated_1D_file_APS(mirrorLength, Step, RandomSeed, SEorFE, RMS)

    return wName_x,wName

def test_2d():
    mirrorLength = 200.0 # mm
    Step = 1.0
    RandomSeed = 898882
    SEorFE = FIGURE_ERROR # 0 = Figure, 1=Slope
    RMS = 1e-7 # mm (0.1 nm)

    mirrorWidth = 10.0
    StepW = 1.0
    RandomSeedW = 7243364
    SEorFEW = FIGURE_ERROR
    RMSW = 1e-8

    x,y,z = create_simulated_2D_profile_APS(mirrorLength, Step, RandomSeed, SEorFE, RMS, mirrorWidth, StepW, RandomSeedW, SEorFEW, RMSW)

    return x,y,z

import os, six
def package_dirname(package):
    """Return the directory path where package is located.

    """
    if isinstance(package, six.string_types):
        package = __import__(package, fromlist=[""])
    filename = package.__file__
    dirname = os.path.dirname(filename)
    return dirname

def test_1d_rrs():
    N = 1000
    mirror_length = 1.0
    height_rms = 3e-9

    profile_type = 1 # 0=Fractal, 1=Gaussian

    if profile_type == 0:
        x, f = create_random_rough_surface_1D(n_surface_points=N,
                                              mirror_length=mirror_length,
                                              rms_height=height_rms,
                                              profile_type=FRACTAL)
    elif profile_type == 1:
        correlation_length = 0.03
        x, f = create_random_rough_surface_1D(n_surface_points=N,
                                              mirror_length=mirror_length,
                                              rms_height=height_rms,
                                              correlation_length=correlation_length,
                                              profile_type=GAUSSIAN)

    return x,f


def test_2d_from_1d():
    values = numpy.loadtxt(package_dirname("srxraylib.metrology") + "/mirror_1d.txt")

    x_coords = values[:, 0]
    y_values = values[:, 1]


    mirrorWidth = 10.0
    StepW = 1.0
    RandomSeedW = 7243364
    SEorFEW = FIGURE_ERROR
    RMSW = 1e-6

    x,y,z = create_2D_profile_from_1D(x_coords, y_values, mirrorWidth, StepW, RandomSeedW, SEorFEW, RMSW)

    return x,y,z

if __name__ == "__main__":
    try:
        from matplotlib import pylab as plt
    except:
        raise ImportError

    test_number = 2 # 0 = all

    if test_number == 1 or test_number == 0:

        wName_x,wName = test_1d()

        f1 = plt.figure(1)
        plt.plot(wName_x,wName)
        plt.title("heights profile")
        plt.xlabel("Y [mm]")
        plt.ylabel("Z [um]")
        plt.show()

    if test_number == 2 or test_number == 0:

        WW_x,SF_x,s = test_2d()

        print(WW_x.size,SF_x.size,s.shape)

        print(WW_x.size,SF_x.size,s.size,WW_x.size*SF_x.size)
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt



        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = numpy.meshgrid(WW_x, SF_x)
        surf = ax.plot_surface(X, Y, s, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        #ax.set_zlim(-1.01, 1.01)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (µm)")

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.06f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    if test_number == 3 or test_number == 0:
        WW_x,SF_x,s = test_2d_from_1d()

        print(WW_x.size,SF_x.size,s.size,WW_x.size*SF_x.size)
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt



        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y = numpy.meshgrid(WW_x, SF_x)
        surf = ax.plot_surface(X, Y, s, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        #ax.set_zlim(-1.01, 1.01)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    if test_number == 4 or test_number == 0:

        wName_x,wName = test_1d_rrs()

        f1 = plt.figure(1)
        plt.plot(wName_x,wName)
        plt.title("heights profile")
        plt.xlabel("Y [mm]")
        plt.ylabel("Z [um]")
        plt.show()

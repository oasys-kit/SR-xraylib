import numpy

def enoise(x):
    tmp = numpy.random.rand()
    return x*tmp

FIGURE_ERROR = 0
SLOPE_ERROR = 1

def create_simulated_1D_file(mirror_length, step, random_seed, error_type, rms):
    """
    :param mirror_length: "Enter mirror length, even"
    :param step: "Step for length "
    :param random_seed: "Random seed between 0 and 1"
    :param error_type: "Figure error (0) or Slope error (1)"
    :param rms: "RMS value of the above"
    :return: x coords, y values
    """
    if(step ==0):
        mirror_length=200.0		#Number of points surface wave
        step=1			#Spacing surface wave
        random_seed=8787
        error_type=0
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
        SF_DIF *= rms / V_sdev
        error_profile *= rms / V_sdev
    elif error_type == FIGURE_ERROR:
        V_sdev = error_profile.std()
        error_profile *= rms / V_sdev

    return error_profile_x, error_profile

def create_simulated_2D_profile(mirror_length, step_l, random_seed_l, error_type_l, rms_l,
                                mirror_width, step_w, random_seed_w, error_type_w, rms_w):
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
        error_type_l=0
        rms_l=0.1e-6
        mirror_width=20		#Number of points surface wave
        step_w=1			#Spacing surface wave
        random_seed_w=0.2
        error_type_w=0
        rms_w=1e-6

    WW_x, WW = create_simulated_1D_file(mirror_width, step_w, random_seed_w, error_type_w, rms_w)
    SF_x, SF = create_simulated_1D_file(mirror_length, step_l, random_seed_l, error_type_l, rms_l)

    npoL = SF.size
    npoW = WW.size

    s = numpy.zeros((npoL, npoW))
    for i in range(npoW):
        s[:,i] = SF + WW[i]

    return WW_x, SF_x, s

def create_2D_profile_from_1D(profile_1D_x, profile_1D_y, mirror_width, step_w, random_seed_w, error_type_w, rms_w):
    numpy.random.seed(seed=random_seed_w)

    if step_w == 0:
        mirror_width=20		#Number of points surface wave
        step_w=1			#Spacing surface wave
        random_seed_w=0.2
        error_type_w=0
        rms_w=1e-6

    WW_x, WW = create_simulated_1D_file(mirror_width, step_w, random_seed_w, error_type_w, rms_w)
    SF_x, SF = profile_1D_x, profile_1D_y

    npoL = SF.size
    npoW = WW.size

    s = numpy.zeros((npoL, npoW))
    for i in range(npoW):
        s[:,i] = SF + WW[i]

    return WW_x, SF_x, s

def test_1d():
    mirrorLength = 200.0 # mm
    Step = 1.0
    RandomSeed = 898882
    SEorFE = FIGURE_ERROR # 0 = Figure, 1=Slope
    RMS = 1e-7 # mm (0.1 nm)
    wName_x,wName = create_simulated_1D_file(mirrorLength, Step, RandomSeed, SEorFE, RMS)

    return wName_x,wName

def test_2d():
    mirrorLength = 200.0 # mm
    Step = 1.0
    RandomSeed = 898882
    SEorFE = FIGURE_ERROR # 0 = Figure, 1=Slope
    RMS = 1e-7 # mm (0.1 nm)

    mirrorWidth = 100.0
    StepW = 1.0
    RandomSeedW = 7243364
    SEorFEW = FIGURE_ERROR
    RMSW = 1e-8

    x,y,z = create_simulated_2D_profile(mirrorLength, Step, RandomSeed, SEorFE, RMS, mirrorWidth, StepW, RandomSeedW, SEorFEW, RMSW)

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

    test_number = 3 # 0 = all

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

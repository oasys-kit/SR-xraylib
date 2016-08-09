import unittest
import numpy


from srxraylib.metrology.profiles_simulation import FIGURE_ERROR,SLOPE_ERROR,GAUSSIAN,FRACTAL
from srxraylib.metrology.profiles_simulation import simulate_profile_1D_gaussian, simulate_profile_1D_fractal
from srxraylib.metrology.profiles_simulation import create_simulated_1D_file_APS
from srxraylib.metrology.profiles_simulation import slopes,simulate_profile_2D, create_simulated_2D_profile_APS
from srxraylib.metrology.profiles_simulation import create_2D_profile_from_1D



do_plot = 0


def package_dirname(package):
    """Return the directory path where package is located.

    """
    import os, six
    if isinstance(package, six.string_types):
        package = __import__(package, fromlist=[""])
    filename = package.__file__
    dirname = os.path.dirname(filename)
    return dirname

#
# 1D tests
#
class ProfilesSimulation1DTest(unittest.TestCase):


    def test_1d_gaussian_figure_error(self):

        mirror_length=200.0
        step=1.0
        rms=1e-7
        correlation_length = 10.0

        x, f = simulate_profile_1D_gaussian(step=step, \
                                              mirror_length=mirror_length, \
                                              rms_heights=rms, \
                                              correlation_length=correlation_length,\
                                              renormalize_to_heights_sd=None)
        slopes = numpy.gradient(f, x[1]-x[0])
        print("test_1d_gaussian: test function: %s, Stdev (not normalized): HEIGHTS=%g.SLOPES=%g"%("test_1d_gaussian_figure_error",f.std(),slopes.std()))

        x, f = simulate_profile_1D_gaussian(step=step, \
                                              mirror_length=mirror_length, \
                                              rms_heights=rms, \
                                              correlation_length=correlation_length,\
                                              renormalize_to_heights_sd=rms)


        print("test_1d_gaussian: test function: %s, HEIGHTS Stdev (normalized to %g)=%g"%("test_1d_gaussian_figure_error",rms,f.std()))
        assert numpy.abs( rms - f.std() ) < 0.01 * numpy.abs(rms)

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(x,f,title="test_1d_gaussian_figure_error",xtitle="Y",ytitle="heights profile Z")

    def test_1d_gaussian_slope_error(self):

        mirror_length=200.0
        step=1.0
        rms_slopes=1.3e-7
        rms_heights = 1e-7
        correlation_length = 10.0

        x, f = simulate_profile_1D_gaussian(step=step, \
                                              mirror_length=mirror_length, \
                                              rms_heights=rms_heights, \
                                              correlation_length=correlation_length,\
                                              renormalize_to_slopes_sd=None)
        slopes = numpy.gradient(f, x[1]-x[0])
        print("test_1d_gaussian: test function: %s, Stdev (not normalized): HEIGHTS=%g.SLOPES=%g"%("test_1d_gaussian_slope_error",f.std(),slopes.std()))

        x, f = simulate_profile_1D_gaussian(step=step, \
                                              mirror_length=mirror_length, \
                                              rms_heights=rms_heights, \
                                              correlation_length=correlation_length,\
                                              renormalize_to_slopes_sd=rms_slopes)
        slopes = numpy.gradient(f, x[1]-x[0])

        print("test_1d_gaussian: test function: %s, SLOPES Stdev (normalized to %g)=%g"%("test_1d_gaussian_slope_error",rms_slopes,slopes.std()))
        assert numpy.abs( rms_slopes - slopes.std() ) < 0.01 * numpy.abs(rms_slopes)

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(x,slopes,title="test_1d_gaussian_slope_error",xtitle="Y",ytitle="slopes Z'")




    def test_1d_fractal_figure_error(self):

        mirror_length=200.0
        step=1.0
        rms=1.25e-7

        x, f = simulate_profile_1D_fractal(step=step,mirror_length=mirror_length,
                                          renormalize_to_heights_sd=rms)
        print("%s, HEIGHTS Stdev: input=%g, obtained=%g"%("test_1d_fractal_figure_error",rms,f.std()))
        assert numpy.abs( rms - f.std() ) < 0.01 * numpy.abs(rms)

        if do_plot:
            from srxraylib.plot.gol import plot
            plot(x,f,title="test_1d_fractal_figure_error",xtitle="Y",ytitle="heights profile Z")

    def test_1d_fractal_slope_error(self):

        mirror_length=200.0
        step=1.0
        rms=3e-6

        x, f = simulate_profile_1D_fractal(step=step,mirror_length=mirror_length,
                                          renormalize_to_slopes_sd=rms)
        slopes = numpy.gradient(f, x[1]-x[0])
        print("%s, SLOPES Stdev: input=%g, obtained=%g"%("test_1d_fractal_slope_error",rms,slopes.std()))
        assert numpy.abs( rms - slopes.std() ) < 0.01 * numpy.abs(rms)
        if do_plot:
            from srxraylib.plot.gol import plot
            plot(x,slopes,title="test_1d_fractal_slope_error",xtitle="Y",ytitle="slopes Z'")


    def test_1d_aps_figure_error(self):

        mirror_length=200.0
        step=1.0
        random_seed=898882
        rms=1e-7

         # units mm
        wName_x,wName = create_simulated_1D_file_APS(mirror_length=mirror_length,step=step, random_seed=random_seed,
                                                     error_type=FIGURE_ERROR, rms=rms)

        print("%s, HEIGHTS Stdev: input=%g, obtained=%g"%("test_1d_aps_figure_error",rms,wName.std()))
        assert numpy.abs( rms - wName.std() ) < 0.01 * numpy.abs(rms)
        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wName_x,wName,title="test_1d_aps_figure_error",xtitle="Y",ytitle="heights profile Z")



    def test_1d_aps_slope_error(self):

        mirror_length=200.0
        step=1.0
        random_seed=898882
        rms=1e-7

         # units mm
        wName_x,wName = create_simulated_1D_file_APS(mirror_length=mirror_length,step=step, random_seed=random_seed,
                                                     error_type=SLOPE_ERROR, rms=rms)

        slopes = numpy.gradient(wName, wName_x[1]-wName_x[0])
        print("test_1d: test function: %s, SLOPES Stdev: input=%g, obtained=%g"%("test_1d_aps_figure_error",rms,slopes.std()))
        assert numpy.abs( rms - slopes.std() ) < 0.01 * numpy.abs(rms)
        if do_plot:
            from srxraylib.plot.gol import plot
            plot(wName_x,slopes,title="test_1d_aps_slope_error",xtitle="Y",ytitle="slopes Z'")


class ProfilesSimulation2DTest(unittest.TestCase):

    def test_2d_normalize_to_figure_error(self,combination="FF"):
        mirrorLength = 200.1 # cm
        Step = 1.0
        RandomSeed = 8788
        RMS = 2.0e-7 # cm


        mirrorWidth = 40.0
        StepW = 1.0
        RandomSeedW = 8788
        RMSW = 1.0e-7

        if combination == "EE": # TODO: not yet tested
            input_file = package_dirname("srxraylib.metrology") + "/mirror_1d.txt"
            values = numpy.loadtxt(input_file)
            x_l = values[:, 0]
            y_l = values[:, 1]
            x_w = values[:, 0]
            y_w = values[:, 1]
            print("File loaded: %s, Length:%f, StDev: %g"%(input_file,x_l[-1]-x_l[0],y_l.std()))
            x,y,z = simulate_profile_2D(random_seed_l=RandomSeed, error_type_l=FIGURE_ERROR, rms_l=RMS,
                                        correlation_length_l=30.0,power_law_exponent_beta_l=1.5,
                                        random_seed_w=RandomSeedW, error_type_w=FIGURE_ERROR, rms_w=RMSW,
                                        correlation_length_w=30.0,power_law_exponent_beta_w=1.5,
                                        x_l=x_l,y_l=y_l,x_w=x_w,y_w=y_w,
                                        combination=combination)
        else:
            x,y,z = simulate_profile_2D(mirror_length=mirrorLength, step_l=Step, random_seed_l=RandomSeed, error_type_l=FIGURE_ERROR, rms_l=RMS,
                                        correlation_length_l=30.0,power_law_exponent_beta_l=1.5,
                                        mirror_width=mirrorWidth, step_w=StepW, random_seed_w=RandomSeedW, error_type_w=FIGURE_ERROR, rms_w=RMSW,
                                        correlation_length_w=30.0,power_law_exponent_beta_w=1.5,
                                        combination=combination)

        tmp1 = slopes(z.T,x,y,return_only_rms=1)

        print("  target HEIGHT error in LENGTH: %g"%(RMS))
        print("  target HEIGHT error in WIDTH: %g"%(RMSW))


        print("  obtained HEIGHT error in LENGTH and WIDTH: %g"%(z.std()))
        print("  obtained SLOPE error in LENGTH: %g"%(tmp1[0]))
        print("  obtained SLOPE error in WIDTH: %g"%(tmp1[1]))

        print("  shape x,y,z:",x.shape,y.shape,z.shape)

        assert numpy.abs( RMS - z.std() ) < 0.01 * numpy.abs(RMS)

        if do_plot:
            from srxraylib.plot.gol import plot_surface
            plot_surface(z.T*1e7,x,y,xtitle="X [cm]",ytitle="Y [cm",ztitle="Z [nm]",title="test_2d_normalize_to_figure_error")


    def test_2d_normalize_to_slope_error(self,combination="FF"):
        mirrorLength = 200.1 # cm
        Step = 1.0
        RandomSeed = 8788
        RMS = 0.2e-6 # 2.0e-6 # rad


        mirrorWidth = 40.0
        StepW = 1.0
        RandomSeedW = 8788
        RMSW = 0.5e-6 # 1.0e-6

        if combination == "EE": # TODO: not yet tested
            input_file = package_dirname("srxraylib.metrology") + "/mirror_1d.txt"
            values = numpy.loadtxt(input_file)
            x_l = values[:, 0]
            y_l = values[:, 1]
            x_w = values[:, 0]
            y_w = values[:, 1]
            print("File loaded: %s, Length:%f, StDev: %g"%(input_file,x_l[-1]-x_l[0],y_l.std()))
            x,y,z = simulate_profile_2D(random_seed_l=RandomSeed, error_type_l=SLOPE_ERROR, rms_l=RMS,
                                        correlation_length_l=30.0,power_law_exponent_beta_l=1.5,
                                        random_seed_w=RandomSeedW, error_type_w=SLOPE_ERROR, rms_w=RMSW,
                                        correlation_length_w=30.0,power_law_exponent_beta_w=1.5,
                                        x_l=x_l,y_l=y_l,x_w=x_w,y_w=y_w,
                                        combination=combination)
        else:
            x,y,z = simulate_profile_2D(mirror_length=mirrorLength, step_l=Step, random_seed_l=RandomSeed, error_type_l=SLOPE_ERROR, rms_l=RMS,
                                        correlation_length_l=30.0,power_law_exponent_beta_l=1.5,
                                        mirror_width=mirrorWidth, step_w=StepW, random_seed_w=RandomSeedW, error_type_w=SLOPE_ERROR, rms_w=RMSW,
                                        correlation_length_w=30.0,power_law_exponent_beta_w=1.5,
                                        combination=combination)

        tmp1 = slopes(z.T,x,y,return_only_rms=1)
        print("  target SLOPE error in WIDTH:  %g rad"%(RMSW))
        print("  target SLOPE error in LENGTH: %g rad"%(RMS))



        print("  obtained HEIGHT error (in LENGTH and WIDTH): %g"%(z.std()))
        print("  obtained SLOPE error in WIDTH: %g"%(tmp1[0]))
        print("  obtained SLOPE error in LENGTH: %g"%(tmp1[1]))

        print("  shape x,y,z:",x.shape,y.shape,z.shape)

        # the LENGTH direction must match!!
        assert numpy.abs( RMS - tmp1[1] ) < 0.01 * numpy.abs(RMS)

        if do_plot:
            from srxraylib.plot.gol import plot_surface
            plot_surface(z.T*1e7,x,y,xtitle="X [cm]",ytitle="Y [cm",ztitle="Z [nm]",title="test_2d_normalize_to_slope_error")




    def test_2d_aps(self):
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

        x,y,z = create_simulated_2D_profile_APS(mirror_length=mirrorLength, step_l=Step, random_seed_l=RandomSeed, \
                                                error_type_l=SEorFE, rms_l=RMS,\
                                                mirror_width=mirrorWidth, step_w=StepW, random_seed_w=RandomSeedW,
                                                error_type_w=SEorFEW, rms_w=RMSW,\
                                                power_law_exponent_beta_l=1.5,power_law_exponent_beta_w=1.5)

        function_name = "create_simulated_2D_profile_APS"



        tmp1 = slopes(z.T,x,y,return_only_rms=1)


        print("test_2d: test function: %s"%(function_name))
        if SEorFE == FIGURE_ERROR:
            print("  target HEIGHT error in LENGTH: %g"%(RMS))
        else:
            print("  target SLOPE error in LENGTH: %g"%(RMS))

        if SEorFEW == FIGURE_ERROR:
            print("  target HEIGHT error in WIDTH: %g"%(RMSW))
        else:
            print("  target SLOPE error in WIDTH: %g"%(RMSW))


        print("  obtained HEIGHT error in LENGTH and WIDTH: %g"%(z.std()))
        print("  obtained SLOPE error in LENGTH: %g"%(tmp1[0]))
        print("  obtained SLOPE error in WIDTH: %g"%(tmp1[1]))


    def test_2d_from_1d_aps(self):
        input_file = package_dirname("srxraylib.metrology") + "/mirror_1d.txt"
        values = numpy.loadtxt(input_file)
        x_coords = values[:, 0]
        y_values = values[:, 1]
        print("File loaded: %s, Length:%f, StDev: %g"%(input_file,x_coords[-1]-x_coords[0],y_values.std()))

        mirrorWidth = 10.0
        StepW = 1.0
        RandomSeedW = 7243364
        SEorFEW = FIGURE_ERROR
        RMSW = 1e-6

        x,y,z = create_2D_profile_from_1D(x_coords, y_values, mirrorWidth, StepW, RandomSeedW, SEorFEW, RMSW)
        function_name = "create_2D_profile_from_1D"
        tmp1 = slopes(z.T,x,y,return_only_rms=1)


        if SEorFEW == FIGURE_ERROR:
            print("  target HEIGHT error in WIDTH: %g"%(RMSW))
        else:
            print("  target SLOPE error in WIDTH: %g"%(RMSW))


        print("  obtained HEIGHT error in LENGTH and WIDTH: %g"%(z.std()))
        print("  obtained SLOPE error in LENGTH: %g"%(tmp1[0]))
        print("  obtained SLOPE error in WIDTH: %g"%(tmp1[1]))
















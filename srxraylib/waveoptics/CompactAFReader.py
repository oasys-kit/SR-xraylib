#
# Propagation of coherent modes in python, then sampling rays for SHADOW
#
# Under development!!
#


import numpy as np

from srxraylib.plot.gol import plot,plot_image,plot_show,plot_scatter
from srxraylib.waveoptics.wavefront2D import Wavefront2D
from srxraylib.waveoptics.propagator2D import propagate_2D_fraunhofer,propagate_2D_fresnel,propagate_2D_fresnel_srw



class CompactAFReader(object):
    def __init__(self, filename):
        file = np.load(filename+".npz")

        self._x_coordinates = file["np_twoform_0"]
        self._y_coordinates = file["np_twoform_1"]
        self._intensity = file["np_twoform_2"]
        self._eigenvalues = file["np_twoform_3"]
        self._modes = np.load(filename+".npy")

    def numberModes(self):
        return self._modes.shape[0]

    def x_coordinates(self):
        return self._x_coordinates

    def y_coordinates(self):
        return self._y_coordinates

    def mode(self, i_mode):
        return self._modes[i_mode,:,:]

    def occupation_number(self, i_mode):
        return self._eigenvalues[i_mode]/np.sum(self._eigenvalues)

    #
    # added srio
    #
    def info(self,verbose=0):
        txt = "***** infor on object: CompactAFReader *****\n"
        txt += "Shape of x:"+repr(self.x_coordinates().shape)+"\n"
        txt += "Shape of y:"+repr(self.y_coordinates().shape)+"\n"
        txt += "Shape of intensity: "+repr(self._intensity.shape)+"\n"
        txt += "Shape of eigenvalues: "+repr(self._eigenvalues.shape)+"\n"
        txt += "Shape of modes: "+repr(self._modes.shape)+"\n\n"


        # txt += ("File %s:" % filename)
        txt += "contains %i modes on the grid \n" % self.numberModes()
        txt += "x: from %e to %e \n" % (self.x_coordinates().min(), self.x_coordinates().max())
        txt += "y: from %e to %e \n" % (self.y_coordinates().min(), self.y_coordinates().max())

        txt += "Occupation and max abs value of the mode\n"
        for i_mode in range(reader.numberModes()):
            occupation = self.occupation_number(i_mode)
            mode = self.mode(i_mode)
            max_abs_value = np.abs(mode).max()
            txt += ("  %i is %e %e \n" % (i_mode, occupation, max_abs_value))

        txt += "********************************************\n"

        if verbose:
            print(txt)

    def get_wavefront2d(self,i_mode,wavelength):
        return Wavefront2D.initialize_wavefront_from_arrays(self.x_coordinates(),self.y_coordinates(),
                                                    self.mode(i_mode),wavelength=wavelength)
#
# tools
#

def line_fwhm(line):
    #
    #CALCULATE fwhm in number of abscissas bins (supposed on a regular grid)
    #
    tt = np.where(line>=max(line)*0.5)
    if line[tt].size > 1:
        # binSize = x[1]-x[0]
        FWHM = (tt[0][-1]-tt[0][0])
        return FWHM
    else:
        return -1

def sample_1d(x,cdf,number_of_points=10000):
    rdm = np.random.rand(number_of_points)

    sampled_points = np.interp(rdm,cdf,x)

    return sampled_points

def sample_rays(x1,y1,mymode,number_of_points=10000):
    II0 = np.abs(mymode.T)
    II0_max = II0.max()
    II0 /= II0_max

    s0 = np.sum(II0,axis=0)
    s1 = np.sum(II0,axis=1)

    s00 = np.cumsum(s0)
    s11 = np.cumsum(s1)

    x = sample_1d(x1,s00/s00[-1],number_of_points=number_of_points)
    y = sample_1d(y1,s11/s11[-1],number_of_points=number_of_points)

    return x,y


def wavefront_intensity_fwhm(wf,prefix="",units="um",verbose=1,shapes=0):
    x = wf.get_coordinate_x()
    y = wf.get_coordinate_y()
    intensity =  wf.get_intensity()

    line_image_h = intensity[:,wf.size()[1]/2]
    line_image_v = intensity[wf.size()[0]/2,:]
    fwhm_h = line_fwhm(line_image_h)* (x[1]-x[0])
    fwhm_v = line_fwhm(line_image_v)* (y[1]-y[0])

    factor = 1.0
    if units == "um":
        factor = 1e6
    if units == "urad":
        factor = 1e6

    if verbose:
        if shapes:
            print("%s Shapes x y z: "%prefix,x.shape,y.shape,wf.get_intensity().shape)
        print("%s FWHM : H: %f %s, V: %f %s"%(prefix,factor*fwhm_h,units,factor*fwhm_v,units, ))

    return fwhm_h,fwhm_v


if __name__ == "__main__":
    #
    # inputs
    #
    filename_source     = "/users/srio/Working/MARKGLASS/new_s3_u18_2m"
    filename_propagated = "/users/srio/Working/MARKGLASS/new_s3_u18_2m_prop_35m"
    wavelength = 1.5533e-10
    mymode_index = 0
    propagation_distance = 35.0

    #
    # Source: retrieve wanted mode for source
    #
    reader = CompactAFReader(filename_source)

    reader.info(verbose=1)

    wf = reader.get_wavefront2d(mymode_index,wavelength)

    # fwhm
    fwhm_h_source,fwhm_v_source = wavefront_intensity_fwhm(wf,prefix="Source wavefront ")
    # plot
    plot_image  (wf.get_phase()**2,1e6*wf.get_coordinate_x(),1e6*reader.y_coordinates(),title="Phases for mode %i"%mymode_index, show=0)


    #
    # propagation
    #

    # method = "fraunhofer"
    # method = "srw"
    method = "fft"

    if method == "fraunhofer":
        wf_rebinned = wf.rebin(4,10,4,15,keep_the_same_intensity=1,set_extrapolation_to_zero=1)
        wf_prop = propagate_2D_fraunhofer(wf_rebinned,propagation_distance=propagation_distance,shift_half_pixel=1)
    elif method == "srw":
        wf_prop = propagate_2D_fresnel_srw(wf,propagation_distance=propagation_distance,srw_autosetting=1)
    elif method == "fft":
        wf_rebinned = wf.rebin(4,10,4,15,keep_the_same_intensity=1,set_extrapolation_to_zero=1)
        plot_image(wf_rebinned.get_phase(), 1e6*wf_rebinned.get_coordinate_x(), 1e6*wf_rebinned.get_coordinate_y(),
                   title="Rebinned phases",show=0)
        number_of_steps = 1
        if number_of_steps == 1:
            wf_prop = propagate_2D_fresnel(wf_rebinned,propagation_distance=propagation_distance,shift_half_pixel=1)
        else:
            wf_prop = wf_rebinned
            for i in range(number_of_steps):
                print(">>> Propagating step %d or %d..."%(i+1,number_of_steps))
                wf_prop = propagate_2D_fresnel(wf_prop,propagation_distance=propagation_distance/number_of_steps,shift_half_pixel=1)
    else:
        raise Exception("Undefined method")


    plot_image(wf_prop.get_intensity(),1e6*wf_prop.get_coordinate_x(),1e6*wf_prop.get_coordinate_y(), show=0,
               title="in-python propagated intensity at 35 m",xtitle="X [um]",ytitle="Y [um]",)

    wavefront_intensity_fwhm(wf_prop,prefix="In-python propagated wavefront",units="urad")

    #
    # get propagated data with SRW (for comparison)
    #
    reader2 = CompactAFReader("/users/srio/Working/MARKGLASS/new_s3_u18_2m_prop_35m")

    wf_prop_srw = reader2.get_wavefront2d(mymode_index,wavelength)

    # plot
    plot_image  (wf_prop_srw.get_intensity(),1e6*wf_prop_srw.get_coordinate_x(),1e6*wf_prop_srw.get_coordinate_y(),
                 title="SRW propagated mode %i"%mymode_index, show=0)
    fwhm_h_srw,fwhm_v_srw = wavefront_intensity_fwhm(wf_prop_srw,prefix="SRW Propagated wavefront")

    #
    # plot profiles
    #
    horizontal_intensity_profile = wf_prop.get_intensity()[:,wf_prop.size()[1]/2]
    horizontal_intensity_profile /= horizontal_intensity_profile.max()
    vertical_intensity_profile = wf_prop.get_intensity()[wf_prop.size()[0]/2,:]
    vertical_intensity_profile /= vertical_intensity_profile.max()

    horizontal_intensity_profile_srw = wf_prop_srw.get_intensity()[:,wf_prop_srw.size()[1]/2]
    horizontal_intensity_profile_srw /= horizontal_intensity_profile_srw.max()
    vertical_intensity_profile_srw =  wf_prop_srw.get_intensity()[wf_prop_srw.size()[0]/2,:]
    vertical_intensity_profile_srw /= vertical_intensity_profile_srw.max()

    plot( 1e6*wf_prop.get_coordinate_x(),     horizontal_intensity_profile,
          1e6*wf_prop_srw.get_coordinate_x(), horizontal_intensity_profile_srw,
          show=0,
          legend=["in-python","SRW"],color=["red","black"],
          title="Horizontal profile of diffracted intensity",xtitle='X [um]',ytitle='Diffracted intensity [a.u.]')


    plot( 1e6*wf_prop.get_coordinate_y(),     vertical_intensity_profile,
          1e6*wf_prop_srw.get_coordinate_y(), vertical_intensity_profile_srw,
          show=0,
          legend=["in-python","SRW"],color=["red","black"],
          title="Vertical profile of diffracted intensity",xtitle='X [um]',ytitle='Diffracted intensity [a.u.]')


    plot_show()

    #
    # sample rays fro SHADOW
    #

    do_shadow = 0

    if do_shadow:

        import Shadow
        npoints = 10000
        xs, ys = sample_rays(wf.get_coordinate_x(),wf.get_coordinate_y(),wf.get_intensity(),number_of_points=npoints)
        xps, yps = sample_rays(wf_prop_srw.get_coordinate_x(),wf_prop_srw.get_coordinate_y(),wf_prop_srw.get_intensity(),
                               number_of_points=npoints)
        #
        plot_scatter(xps,yps,show=1,title="Divergences")

        shadow_beam = Shadow.Beam(10000)
        rays = shadow_beam.rays
        rays[:,10] = 2 * np.pi / (1e2*wavelength)
        rays[:,0] = xs
        rays[:,1] = 0.0
        rays[:,2] = ys
        rays[:,3] = xps
        rays[:,4] = np.sqrt(1 - (xps/propagation_distance)**2 - (yps/propagation_distance)**2)
        rays[:,5] = yps
        rays[:,9] = 1.0
        rays[:,11] = np.arange(1,npoints+1)

        rays[:,6] = 1.0 # electric vector s

        Shadow.ShadowTools.plotxy(shadow_beam,1,3,nbins=100,title="Real space")
        Shadow.ShadowTools.plotxy(shadow_beam,4,6,nbins=100,title="Divergence space")
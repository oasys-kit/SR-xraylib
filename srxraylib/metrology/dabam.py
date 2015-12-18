"""

dabam: (dataBase for metrology)
       python module for processing remote files containing the results
       of metrology measurements on X-ray mirrors

       classes:
             dabam
       main functions:
             cdf (calculate antiderivative function)
             psd (calculate power spectral density)
             write_shadowSurface (writes file with a mesh for SHADOW)
             func_ellipse_slopes evaluates the ellipse slopes profile equation

 
       MODIFICATION HISTORY:
           20130902 srio@esrf.eu, written
           20131109 srio@esrf.eu, added command line arguments, access metadata
           20151103 srio@esrf.eu, restructured to OO
           20151118 srio@esrf.eu, rcleanied and tested

"""

__author__ = "Manuel Sanchez del Rio"
__contact__ = "srio@esrf.eu"
__copyright = "ESRF, 2013-2015"


import numpy
import copy

# to manage input parameters from command-line argument
import sys
import argparse
import json
from io import StringIO
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen


class dabam(object):

    def __init__(self):
        self.description="dabam.py: python program to access and evaluate DAta BAse for Metrology (DABAM) files. See http://ftp.esrf.eu/pub/scisoft/dabam/README.md"

        self.server = "http://ftp.esrf.eu/pub/scisoft/dabam/data/"

        self.inputs = {
            'entryNumber':1,         # 'an integer indicating the DABAM entry number'
            'silent':False,          # 'Silent mode. Default is No'
            'localFileRoot':None,    # 'Define the name of local DABAM file root (<name>.dat for data, <name>.txt for metadata).'
            'outputFileRoot':"",     # 'Define the root for output files. Default is "", so no output files'
            'setDetrending':-2,      # 'Detrending: if >0 is the polynomial degree, -1=skip, -2=automatic, -3=ellipse(optimized) -4=ellipse(design)'
            'nbinS':101,             # 'number of bins of the slopes histogram in rads. '
            'nbinH':101,             # 'number of bins heights histogram in m. '
            'shadowCalc':False,      # 'Write file with mesh for SHADOW.'
            'shadowNy':-1,           # 'For SHADOW file, the number of points along Y (length). If negative, use the profile points. '
            'shadowNx':11,           # 'For SHADOW file, the number of points along X (width). '
            'shadowWidth':6.0,       # 'For SHADOW file, the surface dimension along X (width) in cm. '
            'multiply':1.0,          # 'Multiply input profile (slope or height) by this number (to play with StDev values). '
            'oversample':0.0,        # 'Oversample factor for abscissas. Interpolate profile foor a new one with this factor times npoints'
            'useHeightsOrSlopes':-1, # 'Force calculations using profile heights (0) or slopes (1). Overwrites FILE_FORMAT keyword. Default=-1 (like FILE_FORMAT)'
            'useAbscissasColumn':0,  # 'Use abscissas column index. '
            'useOrdinatesColumn':1,  # 'Use ordinates column index. '
            'plot':None,             # plot data
            'runTests':False,        # run tests cases
            'summary':False,         # get summary of DABAM profiles
            }
        #to load profiles: TODO: rename some variables to more meaningful names
        self.metadata              =  None # metadata
        self.rawdata               =  None # raw datafile
        self.y                     =  None # abscissa along the mirror
        self.zSlopesUndetrended    =  None # undetrended slope profile
        self.zSlopes               =  None # detrended slope profile
        self.zHeightsUndetrended   =  None # undetrended heights profile
        self.zHeights              =  None # detrended heights profile
        self.coeffs                =  None # information on detrending (polynomial coeffs)
        self.f                     =  None # frequency of Power Spectral Density
        self.psdHeights            =  None # Power Spectral Density of Heights profile
        self.psdSlopes             =  None # Power Spectral Density of slopes profile
        self.csdHeights            =  None # Antiderivative of PDF of Heights profile
        self.csdSlopes             =  None # Antiderivative of PDF of Slopes profile
        self.histoSlopes           =  None # to store slopes histogram
        self.histoHeights          =  None # to store heights histogram
        self.momentsSlopes         =  None # to store moments of the slopes profile
        self.momentsHeights        = None # to store moments of the heights profile
        self.powerlaw              = {"hgt_pendent":None, "hgt_shift":None, "slp_pendent":None, "slp_shift":None,
                                "index_from":None,"index_to":None} # to store a dictionary with the results of fitting the PSDs

    #
    #setters (recommended to use setters for changing input and not setting directly the value in self.inputs,
    #         because python does not give errors if the key does not exist but create a new one!)
    #
    def reset(self):
        self.__init__()

    #variables
    def set_input_entryNumber(self,value):
        self.inputs["entryNumber"] = value
    def set_input_silent(self,value):
        self.inputs["silent"] = value
    def set_input_localFileRoot(self,value):
        self.inputs["localFileRoot"] = value
    def set_input_outputFileRoot(self,value):
        self.inputs["outputFileRoot"] = value
    def set_input_setDetrending(self,value):
        self.inputs["setDetrending"] = value
    def set_input_nbinS(self,value):
        self.inputs["nbinS"] = value
    def set_input_nbinH(self,value):
        self.inputs["nbinH"] = value
    def set_input_shadowCalc(self,value):
        self.inputs["shadowCalc"] = value
    def set_input_shadowNy(self,value):
        self.inputs["shadowNy"] = value
    def set_input_shadowNx(self,value):
        self.inputs["shadowNx"] = value
    def set_input_shadowWidth(self,value):
        self.inputs["shadowWidth"] = value
    def set_input_multiply(self,value):
        self.inputs["multiply"] = value
    def set_input_oversample(self,value):
        self.inputs["oversample"] = value
    def set_input_useHeightsOrSlopes(self,value):
        self.inputs["useHeightsOrSlopes"] = value
    def set_input_useAbscissasColumn(self,value):
        self.inputs["useAbscissasColumn"] = value
    def set_input_useOrdinatesColumn(self,value):
        self.inputs["useOrdinatesColumn"] = value
    def set_input_plot(self,value):
        self.inputs["plot"] = value
    def set_input_runTests(self,value):
        self.inputs["runTests"] = value
    def set_input_summary(self,value):
        self.inputs["summary"] = value

    # a shortcut (frequent usage)
    def set_entry(self,value):
        self.inputs["entryNumber"] = value

    #others

    def set_inputs_from_dictionary(self,dict):
        try:
            self.set_input_entryNumber        ( dict["entryNumber"]         )
            self.set_input_silent             ( dict["silent"]              )
            self.set_input_localFileRoot      ( dict["localFileRoot"]       )
            self.set_input_outputFileRoot     ( dict["outputFileRoot"]      )
            self.set_input_setDetrending      ( dict["setDetrending"]       )
            self.set_input_nbinS              ( dict["nbinS"]               )
            self.set_input_nbinH              ( dict["nbinH"]               )
            self.set_input_shadowCalc         ( dict["shadowCalc"]          )
            self.set_input_shadowNy           ( dict["shadowNy"]            )
            self.set_input_shadowNx           ( dict["shadowNx"]            )
            self.set_input_shadowWidth        ( dict["shadowWidth"]         )
            self.set_input_multiply           ( dict["multiply"]            )
            self.set_input_oversample         ( dict["oversample"]          )
            self.set_input_useHeightsOrSlopes ( dict["useHeightsOrSlopes"]  )
            self.set_input_useAbscissasColumn ( dict["useAbscissasColumn"]  )
            self.set_input_useOrdinatesColumn ( dict["useOrdinatesColumn"]  )
            self.set_input_plot               ( dict["plot"]                )
            self.set_input_runTests           ( dict["runTests"]            )
            self.set_input_summary            ( dict["summary"]            )
        except:
            raise Exception("Failed setting dabam input parameters from dictionary")

    #
    # tools
    #
    def is_remote_access(self):
        if (self.get_input_value("localFileRoot") is None):
            remoteAccess = 1  # 0=Local file, 1=Remote file
        else:
            remoteAccess = 0  # 0=Local file, 1=Remote file
        return remoteAccess

    def set_remote_access(self):
        self.set_input_localFileRoot(None)

    #
    #getters
    #

    def get_input_value(self,key):
        try:
            return self.inputs[key]
        except:
            print("****get_input_value: Error returning value for key=%s"%(key))
            return None

    def get_inputs_as_dictionary(self):
        return copy.copy(self.inputs)

    def get_input_value_help(self,key):

        if key == 'entryNumber':        return 'An integer indicating the DABAM entry number or the remote profile files'
        if key == 'silent':             return 'Avoid printing information messages.'
        if key == 'localFileRoot':      return 'Define the name of local DABAM file root (<name>.dat for data, <name>.txt for metadata). If unset, use remore access'
        if key == 'outputFileRoot':     return 'Define the root for output files. Set to "" for no output.  Default is "'+self.get_input_value("outputFileRoot")+'"'
        if key == 'setDetrending':      return 'Detrending: if >0 is the polynomial degree, -1=skip, -2=automatic, -3=ellipse(optimized), -4=ellipse(design). Default=%d'%self.get_input_value("setDetrending")
        if key == 'nbinS':              return 'Number of bins for the slopes histogram in rads. Default is %d'%self.get_input_value("nbinS")
        if key == 'nbinH':              return 'Number of bins for the heights histogram in m. Default is %d'%self.get_input_value("nbinH")
        if key == 'shadowCalc':         return 'Write file with mesh for SHADOW. Default=No'
        if key == 'shadowNy':           return 'For SHADOW file, the number of points along Y (length). If negative, use the profile points. Default=%d'%self.get_input_value("shadowNy")
        if key == 'shadowNx':           return 'For SHADOW file, the number of points along X (width). Default=%d'%self.get_input_value("shadowNx")
        if key == 'shadowWidth':        return 'For SHADOW file, the surface dimension along X (width) in cm. Default=%4.2f'%self.get_input_value("shadowWidth")
        if key == 'multiply':           return 'Multiply input profile (slope or height) by this number (to play with StDev values). Default=%4.2f'%self.get_input_value("multiply")
        if key == 'oversample':         return 'Oversample factor for the number of abscissas points. 0=No oversample. (Default=%2.1f)'%self.get_input_value("oversample")
        if key == 'useHeightsOrSlopes': return 'Force calculations using profile heights (0) or slopes (1). If -1, used metadata keyword FILE_FORMAT. Default=%d'%self.get_input_value("useHeightsOrSlopes")
        if key == 'useAbscissasColumn': return 'Use abscissas column index. Default=%d'%self.get_input_value("useAbscissasColumn")
        if key == 'useOrdinatesColumn': return 'Use ordinates column index. Default=%d'%self.get_input_value("useOrdinatesColumn")
        if key == 'plot':               return 'Plot: all heights slopes psd_h psd_s csd_h csd_s. histo_s histo_h acf_h acf_s. Default=%s'%repr(self.get_input_value("plot"))
        if key == 'runTests':           return 'Run test cases'
        if key == 'summary':            return 'gets a summary of all DABAM profiles'
        return ''


    def get_input_value_short_name(self,key):

        if key == 'entryNumber':         return 'N'
        if key == 'silent':              return 's'
        if key == 'localFileRoot':       return 'l'
        if key == 'outputFileRoot':      return 'r'
        if key == 'setDetrending':       return 'D'
        if key == 'nbinS':               return 'b'
        if key == 'nbinH':               return 'e'
        if key == 'shadowCalc':          return 'S'
        if key == 'shadowNy':            return 'y'
        if key == 'shadowNx':            return 'x'
        if key == 'shadowWidth':         return 'w'
        if key == 'multiply':            return 'm'
        if key == 'oversample':          return 'I'
        if key == 'useHeightsOrSlopes':  return 'Z'
        if key == 'useAbscissasColumn':  return 'A'
        if key == 'useOrdinatesColumn':  return 'O'
        if key == 'plot':                return 'P'
        if key == 'runTests':            return 'T'
        if key == 'summary':             return 'Y'
        return '?'

    #
    # file names
    #
    def file_metadata(self):
        return self._file_root()+'.txt'

    def file_data(self):
        return self._file_root()+'.dat'

    #
    # load profile and store data. This is the main action!!
    #
    def load(self,entry=None):

        if entry is None:
            pass
        else:
            self.set_input_entryNumber(entry)

        # load data and metadata
        self._load_file_metadata()
        self._load_file_data()

        # test consistency
        if (self.get_input_value("localFileRoot") is None):
            if self.get_input_value("entryNumber") <= 0:
                raise Exception("Error: entry number must be non-zero positive for remote access.")

        #calculate detrended profiles
        self._calc_detrended_profiles()

        #calculate psd
        self._calc_psd()

        #calculate histograms
        self._calc_histograms()

        #calculate moments
        self.momentsHeights = moment(self.zHeights)
        self.momentsSlopes = moment(self.zSlopes)

        # write files
        if self.get_input_value("outputFileRoot") != "":
            self._write_output_files()

        #write shadow file
        if self.get_input_value("shadowCalc"):
            self._write_file_for_shadow()
            if not(self.get_input_value("silent")):
                outFile = self.get_input_value("outputFileRoot")+'Shadow.dat'
                print ("File "+outFile+" for SHADOW written to disk.")

        #info
        if not(self.get_input_value("silent")):
            print(self.info_profiles())

    def load_external_profile(self,x,y,type='heights'):
        """
        load a profile from python arrays
        :param x: the abscissas (usually in m)
        :param y: the ordinates (heights in m or slopes in rad)
        :param type: set to 'heights' (default) of 'slopes' depending the type of input
        :return:
        """


        self.set_input_entryNumber(-1)
        self.set_input_multiply(1.0)
        self.set_input_oversample(0.0)
        #self.set_input_setDetrending(-1)
        self.set_input_useHeightsOrSlopes(0)
        self.set_input_localFileRoot("<none (from python variable)>")
        self.rawdata = numpy.vstack((x,y)).T
        self.metadata = {}
        self.metadata["SURFACE_SHAPE"] = ""
        self.metadata["FACILITY"] = ""
        self.metadata["CALC_SLOPE_RMS"] = None
        self.metadata["CALC_HEIGHT_RMS"] = None

        self.metadata["X1_FACTOR"] = 1.0
        self.metadata["Y1_FACTOR"] = 1.0

        self.set_input_useAbscissasColumn(0)
        self.set_input_useOrdinatesColumn(1)

        if type == 'heights':
            self.metadata["FILE_FORMAT"] = 2
            self.set_input_useHeightsOrSlopes(0)
        elif type == 'slopes':
            self.metadata["FILE_FORMAT"] = 1
            self.set_input_useHeightsOrSlopes(1)
        else:
            raise Exception("dabam.load_external_profile: Unknown type")


        #calculate detrended profiles
        self._calc_detrended_profiles()

        #calculate psd
        self._calc_psd()

        #calculate histograms
        self._calc_histograms()

        #calculate moments
        self.momentsHeights = moment(self.zHeights)
        self.momentsSlopes = moment(self.zSlopes)

        # write files
        if self.get_input_value("outputFileRoot") != "":
            self._write_output_files()

        #write shadow file
        if self.get_input_value("shadowCalc"):
            self._write_file_for_shadow()
            if not(self.get_input_value("silent")):
                outFile = self.get_input_value("outputFileRoot")+'Shadow.dat'
                print ("File "+outFile+" for SHADOW written to disk.")

        #info
        if not(self.get_input_value("silent")):
            print(self.info_profiles())



    #
    #calculations
    #

    def stdev_profile_heights(self):
        return self.zHeights.std(ddof=1)

    def stdev_profile_slopes(self):
        return self.zSlopes.std(ddof=1)

    def stdev_psd_heights(self):
        return numpy.sqrt(self.csdHeights[-1])

    def stdev_psd_slopes(self):
        return numpy.sqrt(self.csdSlopes[-1])

    def stdev_user_heights(self):
        if self.metadata['CALC_HEIGHT_RMS'] != None:
            if self.metadata['CALC_HEIGHT_RMS_FACTOR'] != None:
                return float(self.metadata['CALC_HEIGHT_RMS']) * float(self.metadata['CALC_HEIGHT_RMS_FACTOR'])
            else:
                return float(self.metadata['CALC_HEIGHT_RMS'])

    def stdev_user_slopes(self):
       if self.metadata['CALC_SLOPE_RMS'] != None:
            if self.metadata['CALC_SLOPE_RMS_FACTOR'] != None:
                return float(self.metadata['CALC_SLOPE_RMS']) * float(self.metadata['CALC_SLOPE_RMS_FACTOR'])
            else:
                return float(self.metadata['CALC_SLOPE_RMS'])

    def csd_heights(self):
        return numpy.sqrt(self.csdHeights)/self.stdev_psd_heights()

    def csd_slopes(self):
        return numpy.sqrt(self.csdSlopes)/self.stdev_psd_slopes()

    def autocorrelation_heights(self):
        c1,c2,c3 = autocorrelationfunction(self.y,self.zHeights)
        return c3

    def autocorrelation_slopes(self):
        c1,c2,c3  = autocorrelationfunction(self.y,self.zSlopes)
        return c3
    #
    # info
    #
    def info_profiles(self):

        if self.zHeights is None:
            return "Error: no loaded profile."

        txt = ""

        if int(self.get_input_value("setDetrending")) == -2: # this is the default
            if (self.metadata['SURFACE_SHAPE']).lower() == "elliptical":
                polDegree = -3     # elliptical detrending
            else:
                polDegree = 1      # linear detrending
        else:
            polDegree = int(self.get_input_value("setDetrending"))


        #;
        #; info
        #;
        #
        txt += '\n---------- profile results -------------------------\n'
        if (self.get_input_value("localFileRoot") is None):
            txt += 'Remote directory:\n   %s\n'%self.server
        txt += 'Data File:     %s\n'%self.file_data()
        txt += 'Metadata File: %s\n'%self.file_metadata()
        txt += 'Surface shape: %s\n'%(self.metadata['SURFACE_SHAPE'])
        txt += 'Facility:      %s\n'%(self.metadata['FACILITY'])
        txt += 'Scan length: %.3f mm\n'%(1e3*(self.y[-1]-self.y[0]))
        txt += 'Number of points: %d\n'%(len(self.y))

        txt += '\n'

        if polDegree >= 0:
            if polDegree == 1:
                txt += "Linear detrending: z'=%g x%+g"%(self.coeffs[0],self.coeffs[1])+"\n"
                txt += 'Radius of curvature: %.3F m'%(1.0/self.coeffs[-2])+"\n"
            else:
                txt += 'Polynomial detrending coefficients: '+repr(self.coeffs)+"\n"
        elif polDegree == -1:
           txt += 'No detrending applied.\n'
        elif polDegree == -3:
           txt += 'Ellipse detrending applied. Using Optimized parameters:\n'
           txt += '         p = %f m \n'%self.coeffs[0]
           txt += '         q = %f m \n'%self.coeffs[1]
           txt += '         theta = %f rad \n'%self.coeffs[2]
           txt += '         vertical shift = %f nm \n'%self.coeffs[3]
        elif polDegree == -4:
           txt += 'Ellipse detrending applied. Usinng Design parameters:\n'
           txt += '         p = %f m \n'%self.coeffs[0]
           txt += '         q = %f m \n'%self.coeffs[1]
           txt += '         theta = %f rad \n'%self.coeffs[2]
           txt += '         vertical shift = %f nm \n'%self.coeffs[3]

        txt += self.statistics_summary()

        txt += '----------------------------------------------------\n'
        return txt

    def statistics_summary(self):
        txt = ""
        txt += 'Slopes profile:\n'
        txt += '         StDev of slopes profile:    %.3f urad\n' %( 1e6*self.stdev_profile_slopes() )
        txt += '         from PSD:                   %.3f urad\n' %( 1e6*self.stdev_psd_slopes())
        if self.stdev_user_slopes() != None:
            txt += '         from USER (metadata):       %.3f urad\n'   %(1e6*self.stdev_user_slopes())
        txt += '         Peak-to-valley: no detrend: %.3f urad\n'   %(1e6*(self.zSlopesUndetrended.max() - self.zSlopesUndetrended.min()))
        txt += '                       with detrend: %.3f urad\n'   %(1e6*(self.zSlopes.max() - self.zSlopes.min() ))
        txt += '         Skewness: %.3f, Kurtosis: %.3f\n'   %(self.momentsSlopes[2],self.momentsSlopes[3])
        beta = -self.powerlaw["slp_pendent"]
        txt += '         PSD power law fit: beta:%.3f, Df: %.3f\n'   %(beta,(5-beta)/2)
        txt += '         Autocorrelation length:%.3f\n'   %(self.autocorrelation_slopes())

        txt += 'Heights profile: \n'
        txt += '         StDev of heights profile:   %.3f nm\n'   %(1e9*self.stdev_profile_heights() )
        txt += '         from PSD:                   %.3f nm\n'   %(1e9*self.stdev_psd_heights() )
        if self.stdev_user_heights() != None:
            txt += '         from USER (metadata):       %.3f nm\n'   %(1e9*self.stdev_user_heights())
        txt += '         Peak-to-valley: no detrend: %.3f nm\n'   %(1e9*(self.zHeightsUndetrended.max() - self.zHeightsUndetrended.min()))
        txt += '                       with detrend: %.3f nm\n'   %(1e9*(self.zHeights.max() - self.zHeights.min() ))
        txt += '         Skewness: %.3f, Kurtosis: %.3f\n'   %(self.momentsHeights[2],self.momentsHeights[3])
        beta = -self.powerlaw["hgt_pendent"]
        txt += '         PSD power law fit: beta:%.3f, Df: %.3f\n'   %(beta,(5-beta)/2)
        txt += '         Autocorrelation length:%.3f\n'   %(self.autocorrelation_heights())

        return txt

    def plot(self):
        try:
            from matplotlib import pylab as plt
        except:
            print("Cannot make plots. Please install matplotlib.")
            return None

        what = self.get_input_value("plot")

        if what == "all":
            what = ["heights","slopes","psd_h","psd_s","csd_h","cds_s","histo_s","histo_h"]
        else:
            what = what.split(" ")

        for i,iwhat in enumerate(what):
            print("plotting: ",iwhat)
            if (iwhat == "heights" ):
                f1 = plt.figure(1)
                plt.plot(1e3*self.y,1e6*self.zHeights)
                plt.title("heights profile")
                plt.xlabel("Y [mm]")
                plt.ylabel("Z [um]")
            elif (iwhat == "slopes"):
                f2 = plt.figure(2)
                plt.plot(1e3*self.y,1e6*self.zSlopes)
                plt.title("slopes profile")
                plt.xlabel("Y [mm]")
                plt.ylabel("Zp [urad]")
            elif (iwhat == "psd_h"):
                f3 = plt.figure(3)
                plt.loglog(self.f,self.psdHeights)
                y = self.f**(self.powerlaw["hgt_pendent"])*10**self.powerlaw["hgt_shift"]
                i0 = self.powerlaw["index_from"]
                i1 = self.powerlaw["index_to"]
                plt.loglog(self.f,y)
                plt.loglog(self.f[i0:i1],y[i0:i1])
                beta = -self.powerlaw["hgt_pendent"]
                plt.title("PSD of heights profile (beta=%.2f,Df=%.2f)"%(beta,(5-beta)/2))
                plt.xlabel("f [m^-1]")
                plt.ylabel("PSD [m^3]")
            elif (iwhat == "psd_s"):
                f4 = plt.figure(4)
                plt.loglog(self.f,self.psdSlopes)
                y = self.f**(self.powerlaw["slp_pendent"])*10**self.powerlaw["slp_shift"]
                i0 = self.powerlaw["index_from"]
                i1 = self.powerlaw["index_to"]
                plt.loglog(self.f,y)
                plt.loglog(self.f[i0:i1],y[i0:i1])
                beta = -self.powerlaw["slp_pendent"]
                plt.title("PSD of slopes profile (beta=%.2f,Df=%.2f)"%(beta,(5-beta)/2))
                plt.xlabel("f [m^-1]")
                plt.ylabel("PSD [rad^3]")
            elif (iwhat == "csd_h"):
                f5 = plt.figure(5)
                plt.semilogx(self.f,self.csd_heights())
                plt.title("Cumulative Spectral Density of heights profile")
                plt.xlabel("f [m^-1]")
                plt.ylabel("csd_h")
            elif (iwhat == "csd_s"):
                f6 = plt.figure(6)
                plt.semilogx(self.f,self.csd_slopes())
                plt.title("Cumulative Spectral Density  of slopes profile")
                plt.xlabel("f [m^-1]")
                plt.ylabel("csd_s")
            elif (iwhat == "histo_s" ):
                f7 = plt.figure(7)
                plt.plot(1e6*self.histoSlopes["x_path"],self.histoSlopes["y1_path"])
                plt.plot(1e6*self.histoSlopes["x_path"],self.histoSlopes["y2_path"])
                plt.title("slopes histogram and Gaussian with StDev: %10.3f urad"%(1e6*self.stdev_profile_slopes()))
                plt.xlabel("Z' [urad]")
                plt.ylabel("counts")
            elif (iwhat == "histo_h" ):
                f8 = plt.figure(8)
                plt.plot(1e9*self.histoHeights["x_path"],self.histoHeights["y1_path"])
                plt.plot(1e9*self.histoHeights["x_path"],self.histoHeights["y2_path"])
                plt.title("heights histogram and Gaussian with StDev: %10.3f nm"%(1e9*self.stdev_profile_heights()))
                plt.xlabel("Z [nm]")
                plt.ylabel("counts")
            elif (iwhat == "acf_h" ):
                f9 = plt.figure(9)
                c1,c2,c3 = autocorrelationfunction(self.y,self.zHeights)
                plt.plot(c1[0:-1],c2)
                plt.title("Heights autocovariance. Autocorrelation length (acf_h=0.5)=%.3f m"%(c3))
                plt.xlabel("Length [m]")
                plt.ylabel("acf")
            elif (iwhat == "acf_s" ):
                f10 = plt.figure(10)
                c1,c2,c3 = autocorrelationfunction(self.y,self.zSlopes)
                plt.plot(c1[0:-1],c2)
                plt.title("Slopes autocovariance. Autocorrelation length (acf_s=0.5)=%.3f m"%(c3))
                plt.xlabel("Length [m]")
                plt.ylabel("acf_s")
            else:
                print("Plotting options are: heights slopes psd_h psd_s csd_h csd_s acf_h acf_s")
                return None
        plt.show()


    #
    # auxiliar methods for internal use
    #
    def _set_from_command_line(self):
        #
        # define default aparameters taken from command arguments
        #
        parser = argparse.ArgumentParser(description=self.description)

        # main argument

        parser.add_argument('entryNumber', nargs='?', metavar='N', type=int, default=self.get_input_value('entryNumber'),
            help=self.get_input_value_help('entryNumber'))

        parser.add_argument('-'+self.get_input_value_short_name('runTests'), '--runTests', action='store_true',
            help=self.get_input_value_help('runTests'))

        parser.add_argument('-'+self.get_input_value_short_name('summary'), '--summary', action='store_true',
            help=self.get_input_value_help('summary'))

        # options (flags)

        parser.add_argument('-'+self.get_input_value_short_name('silent'),'--silent', action='store_true', help=self.get_input_value_help('silent'))

        #options (parameters)

        parser.add_argument('-'+self.get_input_value_short_name('localFileRoot'), '--localFileRoot', help=self.get_input_value_help('localFileRoot'))

        parser.add_argument('-'+self.get_input_value('outputFileRoot'), '--outputFileRoot', default=self.get_input_value('outputFileRoot'),
            help=self.get_input_value_help('outputFileRoot'))

        parser.add_argument('-'+self.get_input_value_short_name('setDetrending'), '--setDetrending', default=self.get_input_value('setDetrending'),
            help=self.get_input_value_help('setDetrending'))

        parser.add_argument('-'+self.get_input_value_short_name('nbinS'), '--nbinS', default=self.get_input_value('nbinS'),
            help=self.get_input_value_help('nbinS'))

        parser.add_argument('-'+self.get_input_value_short_name('nbinH'), '--nbinH', default=self.get_input_value('nbinH'),
            help=self.get_input_value_help('nbinH'))

        parser.add_argument('-'+self.get_input_value_short_name('shadowCalc'), '--shadowCalc', action='store_true',
            help=self.get_input_value_help('shadowCalc'))

        parser.add_argument('-'+self.get_input_value_short_name('shadowNy'), '--shadowNy', default=self.get_input_value('shadowNy'),
            help=self.get_input_value_help('shadowNy'))
        parser.add_argument('-'+self.get_input_value_short_name('shadowNx'), '--shadowNx', default=self.get_input_value('shadowNx'),
            help=self.get_input_value_help('shadowNx'))

        parser.add_argument('-'+self.get_input_value_short_name('shadowWidth'), '--shadowWidth', default=self.get_input_value('shadowWidth'),
            help=self.get_input_value_help('shadowWidth'))

        parser.add_argument('-'+self.get_input_value_short_name('multiply'), '--multiply', default=self.get_input_value('multiply'),
            help=self.get_input_value_help('multiply'))

        parser.add_argument('-'+self.get_input_value_short_name('oversample'), '--oversample', default=self.get_input_value('oversample'),
            help=self.get_input_value_help('oversample'))

        parser.add_argument('-'+self.get_input_value_short_name('useHeightsOrSlopes'), '--useHeightsOrSlopes', default=self.get_input_value('useHeightsOrSlopes'),
            help=self.get_input_value_help('useHeightsOrSlopes'))

        parser.add_argument('-'+self.get_input_value_short_name('useAbscissasColumn'), '--useAbscissasColumn', default=self.get_input_value('useAbscissasColumn'),
            help=self.get_input_value_help('useAbscissasColumn'))

        parser.add_argument('-'+self.get_input_value_short_name('useOrdinatesColumn'), '--useOrdinatesColumn', default=self.get_input_value('useOrdinatesColumn'),
            help=self.get_input_value_help('useOrdinatesColumn'))

        parser.add_argument('-'+self.get_input_value_short_name('plot'), '--plot', default=self.get_input_value('plot'),
            help=self.get_input_value_help('plot'))


        args = parser.parse_args()

        self.set_input_entryNumber(args.entryNumber)
        self.set_input_silent(args.silent)
        self.set_input_localFileRoot(args.localFileRoot)
        self.set_input_outputFileRoot(args.outputFileRoot)
        self.set_input_setDetrending(args.setDetrending)
        self.set_input_nbinS(args.nbinS)
        self.set_input_nbinH(args.nbinH)
        self.set_input_shadowCalc(args.shadowCalc)
        self.set_input_shadowNy(args.shadowNy)
        self.set_input_shadowNx(args.shadowNx)
        self.set_input_shadowWidth(args.shadowWidth)
        self.set_input_multiply(args.multiply)
        self.set_input_oversample(args.oversample)
        self.set_input_useHeightsOrSlopes(args.useHeightsOrSlopes)
        self.set_input_useAbscissasColumn(args.useAbscissasColumn)
        self.set_input_useOrdinatesColumn(args.useOrdinatesColumn)
        self.set_input_plot(args.plot)
        self.set_input_runTests(args.runTests)
        self.set_input_summary(args.summary)

    def _file_root(self):

        if self.is_remote_access():
            input_option = self.get_input_value("entryNumber")
            inFileRoot = "dabam-"+"%03d"%(input_option)
        else:
            inFileRoot = self.get_input_value("localFileRoot")

        return inFileRoot

    def _load_file_metadata(self):
        if self.is_remote_access():
            # metadata file
            myfileurl = self.server+self.file_metadata()
            u = urlopen(myfileurl)
            ur = u.read()
            ur1 = ur.decode(encoding='UTF-8')
            h = json.loads(ur1) # dictionnary with metadata
            self.metadata = h
        else:
            try:
                with open(self.file_metadata(), mode='r') as f1:
                    h = json.load(f1)
                self.metadata = h
            except:
                print ("_load_file_metadata: Error accessing local file: "+self.file_metadata())


    def _load_file_data(self):
        if self.is_remote_access():
            # data
            myfileurl = self.server+self.file_data()
            try:
                u = urlopen(myfileurl)
            except:
                print ("_load_file_data: Error accessing remote file: "+myfileurl+" does not exist.")
                return None

            ur = u.read()

            skipLines = self.metadata['FILE_HEADER_LINES']

            if sys.version_info[0] == 2:
                ur = StringIO(unicode(ur))
            else:
                ur = StringIO(ur.decode(encoding='ASCII'))

            a = numpy.loadtxt(ur, skiprows=skipLines )
            self.rawdata = a
        else:
            try:
                skipLines = self.metadata['FILE_HEADER_LINES']
                import os
                a = numpy.loadtxt(self.file_data(), skiprows=skipLines) #, dtype="float64" )
                self.rawdata = a
            except:
                print ("Error accessing local file: "+self.file_data())



    def _calc_detrended_profiles(self):
        """
        Retrieve detrended profiles (slope and height): abscissa slope slope_detrended heights heights_detrended
        :return:
        """

        #;
        #; convert to SI units (m,rad)
        #;
        a = self.rawdata.copy()

        a[:,0] = a[:,0]*self.metadata['X1_FACTOR']
        a[:,1] = a[:,1]*self.metadata['Y1_FACTOR']
        ncols = a.shape[1]

        if int(self.metadata["FILE_FORMAT"]) <= 2:
            for i in range(2,ncols):    # X1 Y1 Y2 Y3...
                a[:,i] = a[:,i]*self.metadata['Y%d_FACTOR'%i]
        elif int(self.metadata["FILE_FORMAT"]) == 3: #X1 Y1 X2 Y2 etc
            ngroups = int(ncols / 2)
            icol = 1
            for i in range(2,ngroups):    # X1 Y1 Y2 Y3...
                icol += 1
                a[:,icol] = a[:,icol]*self.metadata['X%d_FACTOR'%i]
                icol += 1
                a[:,icol] = a[:,icol]*self.metadata['Y%d_FACTOR'%i]

        #
        #; apply multiplicative factor
        #
        if (self.get_input_value("multiply") != 1.0):
            factor = float(self.get_input_value("multiply"))
            a[:,1] = a[:,1]  * factor
            if not(self.get_input_value("silent")):
                print("Multiplicative factor %.3f applied."%(factor))






        #
        # select columns with abscissas and ordinates
        #
        col_abscissas = int( self.get_input_value("useAbscissasColumn") )
        col_ordinates = int( self.get_input_value("useOrdinatesColumn") )

        col_ordinates_title = 'unknown'
        if self.metadata['FILE_FORMAT'] == 1:  # slopes in Col2
            col_ordinates_title = 'slopes'
        if self.metadata['FILE_FORMAT'] == 2:  # heights in Col2
            col_ordinates_title = 'heights'
        if self.metadata['FILE_FORMAT'] == 3:  # slopes in Col2, file X1 Y1 X2 Y2
            col_ordinates_title = 'slopes'
        if self.metadata['FILE_FORMAT'] == 4:  # heights in Col2, file X1 Y1 X2 Y2
            col_ordinates_title = 'heights'

        if int(self.get_input_value("useHeightsOrSlopes")) == -1:  #default, keep current
            pass
        else: # overwrite
            if int(self.get_input_value("useHeightsOrSlopes")) == 0:
                col_ordinates_title = 'heights'
            if int(self.get_input_value("useHeightsOrSlopes")) == 1:
                col_ordinates_title = 'slopes'

        if not(self.get_input_value("silent")):
            print("Using: abscissas column index %d (mirror coordinates)"%(col_abscissas))
            print("       ordinates column index %d (profile %s)"%(col_ordinates,col_ordinates_title))

        #;
        #; Extract right columns and interpolate (if wanted)
        #; substract linear fit to the slopes (remove best circle from profile)
        #;

        a_h = a[:,col_abscissas]
        a_v = a[:,col_ordinates]

        factor = float(self.get_input_value("oversample"))
        if (factor > 1e-6):
            npoints = a_h.size
            npoints1 = int(npoints * factor)
            a_hi = numpy.linspace(a_h.min(),a_h.max(),npoints1)
            a_vi = numpy.interp(a_hi,a_h,a_v)
            a_h = a_hi
            a_v = a_vi
            if not(self.get_input_value("silent")):
                print("Oversampling/interpolating from %d to %d points."%(npoints,npoints1))

        if col_ordinates_title == 'slopes':
            sy = a_h
            sz1 = a_v
        elif col_ordinates_title == 'heights':
            sy = a_h
            #TODO we suppose that data are equally spaced. Think how to generalise
            sz1 = numpy.gradient(a_v,(sy[1]-sy[0]))
        else:
            raise NotImplementedError


        #;
        #; Detrending:
        #; substract linear fit to the slopes (remove best circle from profile)
        #;
        sz = numpy.copy(sz1)

        # define detrending to apply: >0 polynomial prder, -1=None, -2=Default, -3=elliptical

        if int(self.get_input_value("setDetrending")) == -2: # this is the default
            if (self.metadata['SURFACE_SHAPE']).lower() == "elliptical":
                polDegree = -3     # elliptical detrending
            else:
                polDegree = 1      # linear detrending
        else:
            polDegree = int(self.get_input_value("setDetrending"))


        if polDegree >= 0: # polinomial fit
            coeffs = numpy.polyfit(sy, sz1, polDegree)
            pol = numpy.poly1d(coeffs)
            zfit = pol(sy)
            sz = sz1 - zfit
        else:
            coeffs = None

        if polDegree == -3: # ellipse (optimized)
            coeffs = None
            try:
                from scipy.optimize import curve_fit, leastsq
            except:
                raise ImportError("Cannot perform ellipse detrending: please install scipy")

            if not(self.get_input_value("silent")):
                print("Detrending an ellipse...")
            if ("ELLIPSE_DESIGN_P" in self.metadata) and ("ELLIPSE_DESIGN_Q" in self.metadata) and ("ELLIPSE_DESIGN_THETA" in self.metadata):
                ell_p = self.metadata["ELLIPSE_DESIGN_P"]
                ell_q = self.metadata["ELLIPSE_DESIGN_Q"]
                ell_theta = self.metadata["ELLIPSE_DESIGN_THETA"]

                fitfunc_ell_slopes  =  lambda p, x: func_ellipse_slopes(x, p[0], p[1], p[2], p[3])

                errfunc_ell_slopes = lambda p, x, y: fitfunc_ell_slopes(p, x) - y

                p_guess = [ell_p,ell_q,ell_theta,0.0]

                szGuess = fitfunc_ell_slopes(p_guess, sy)

                coeffs, cov_x, infodic, mesg, ier = leastsq(errfunc_ell_slopes, p_guess, args=(sy, sz1), full_output=True)


                #zpopt= func_ellipse_slopes(sy, popt[0], popt[1], popt[2], popt[3])
                szOptimized  = fitfunc_ell_slopes(coeffs, sy)
                sz = sz1 - szOptimized

                if not(self.get_input_value("silent")):
                    print("Ellipse design parameters found in metadata: p=%f m,q=%f m,theta=%f rad, shift=%f nm, Slopes_Std=%f urad"%
                          (ell_p,ell_q,ell_theta,0.0,1e6*(sz1-szGuess).std(ddof=1) ))
                    print("Optimized ellipse                          : p=%f m,q=%f m,theta=%f rad, shift=%f nm, Slopes_Std=%f urad\n"%
                          (coeffs[0],coeffs[1],coeffs[2],coeffs[3],1e6*sz.std(ddof=1) ))
            else:
                if not(self.get_input_value("silent")):
                    print("Ellipse design parameters NOT FOUND in metadata. Guessing parameters (may be unrealistic!)")
                coeffs, cov_x = curve_fit(func_ellipse_slopes, sy, sz1, maxfev=10000)
                szOptimized= func_ellipse_slopes(sy, coeffs[0], coeffs[1], coeffs[2], coeffs[3])
                sz = sz1 - szOptimized

        if polDegree == -4: # ellipse (design)
            if not(self.get_input_value("silent")):
                print("Detrending an ellipse...")
            if ("ELLIPSE_DESIGN_P" in self.metadata) and ("ELLIPSE_DESIGN_Q" in self.metadata) and ("ELLIPSE_DESIGN_THETA" in self.metadata):
                coeffs = numpy.zeros(4)
                coeffs[0] = self.metadata["ELLIPSE_DESIGN_P"]
                coeffs[1] = self.metadata["ELLIPSE_DESIGN_Q"]
                coeffs[2] = self.metadata["ELLIPSE_DESIGN_THETA"]
                coeffs[3] = 0.0
                fitfunc_ell_slopes  =  lambda p, x: func_ellipse_slopes(x, p[0], p[1], p[2], p[3])
                szGuess = fitfunc_ell_slopes(coeffs, sy)
                sz = sz1 - szGuess
            else:
                print("Error: Ellipse detrend parameters not found in metadata")
                raise RuntimeError


        #;
        #; calculate heights by integrating the slope
        #;
        zprof = cdf(sy,sz)
        zprof1 = cdf(sy,sz1)

        self.y = sy
        self.zSlopesUndetrended = sz1
        self.zSlopes = sz
        self.zHeightsUndetrended = zprof1
        self.zHeights = zprof
        self.coeffs = coeffs


    def _calc_psd(self):
        sy    = self.y
        #sz1    = self.sz1
        sz    = self.zSlopes
        #zprof1    = self.zprof1
        zprof     = self.zHeights

        #;
        #; calculate PSD on both profile and slope, and also then their antiderivative
        #;
        psdHeights,f = psd(sy,zprof,onlyrange=None)
        psdSlopes,f = psd(sy,sz,onlyrange=None)
        adpsdHeights = cdf(f,psdHeights)
        adpsdSlopes  = cdf(f,psdSlopes)



        self.f = f
        self.psdHeights   = psdHeights
        self.psdSlopes    = psdSlopes
        self.csdHeights = adpsdHeights
        self.csdSlopes  = adpsdSlopes

        #fit PSD to a power law
        x = numpy.log10(self.f)
        y_h = numpy.log10(self.psdHeights)
        y_s = numpy.log10(self.psdSlopes)
        #select the fitting area (80% of the full interval, centered)

        x_left = (x.min()+0.1*(x.max()-x.min()))
        x_right = (x.max()-0.1*(x.max()-x.min()))

        # redefine  left limit for the fit to the frequency value corresponding to the correlation length
        # acf_h = autocovariance_1D(self.sy,self.zprof)
        # f_h = numpy.log10( 1.0 / acf_h[2] )
        # x_left = f_h

        c1 = (x < x_right )
        c2 = (x > x_left )
        igood = numpy.where(c1 & c2)
        igood = numpy.array(igood)
        igood.shape = -1

        coeffs_h = numpy.polyfit(x[igood], y_h[igood], 1)
        coeffs_s = numpy.polyfit(x[igood], y_s[igood], 1)

        self.powerlaw = {"hgt_pendent":coeffs_h[0], "hgt_shift":coeffs_h[1], \
                         "slp_pendent":coeffs_s[0], "slp_shift":coeffs_s[1],\
                         "index_from":igood[0],"index_to":igood[-1]}


    def _calc_histograms(self):
        """
        Calculates slopes and heights histograms and also the Gaussians with their StDev

        results are stored in:
        self.histoSlopes = {"x":hy_center, "y1":hz, "y2":g, "x_path":hy_path, "y1_path":hz_path, "y2_path":g_path}

        where:
          x is the abscissas (at bin center), y1 is the histogram, y2 is the Gaussian
          x_path is the abscissas with points at left and riggh edges of each bin, y1_path is the
        :return:
        """

        #
        # slopes histogram
        #

        # binsize = float(self.get_input_value("binS")) # default is 1e-7 rads
        # bins = numpy.ceil( (self.sz.max()-self.sz.min())/binsize )

        bins = int(self.get_input_value("nbinS"))
        hz,hy_left = numpy.histogram(self.zSlopes, bins = bins)


        hy_center = hy_left[0:-1]+0.5*(hy_left[1]-hy_left[0]) #calculate positions of the center of the bins
        hy_right  = hy_left[0:-1]+1.0*(hy_left[1]-hy_left[0]) #calculate positions of the right edge of the bins

        hy_path = []
        hz_path = []
        for s,t,v in zip(hy_left,hy_right,hz):
            hy_path.append(s)
            hz_path.append(v)
            hy_path.append(t)
            hz_path.append(v)

        hy_path = numpy.array(hy_path)
        hz_path = numpy.array(hz_path)

        #Gaussian with StDev of data
        g = numpy.exp( -numpy.power(hy_center-self.zSlopes.mean(),2)/2/numpy.power(self.stdev_profile_slopes(),2) )
        g = g/g.sum()*hz.sum()

        g_path = numpy.exp( -numpy.power(hy_path-self.zSlopes.mean(),2)/2/numpy.power(self.stdev_profile_slopes(),2) )
        g_path = g_path/g_path.sum()*hz_path.sum()


        self.histoSlopes = {"x":hy_center, "y1":hz, "y2":g, "x_path":hy_path, "y1_path":hz_path, "y2_path":g_path}

        #
        # heights histogram
        #

        # binsize = float(self.get_input_value("binH"))
        # bins = numpy.ceil( (self.zprof.max()-self.zprof.min())/binsize )
        bins = int(self.get_input_value("nbinH"))
        hz,hy_left = numpy.histogram(self.zHeights, bins = bins)

        hy_center = hy_left[0:-1]+0.5*(hy_left[1]-hy_left[0]) #calculate positions of the center of the bins
        hy_right  = hy_left[0:-1]+1.0*(hy_left[1]-hy_left[0]) #calculate positions of the right edge of the bins

        hy_path = []
        hz_path = []
        for s,t,v in zip(hy_left,hy_right,hz):
            hy_path.append(s)
            hz_path.append(v)
            hy_path.append(t)
            hz_path.append(v)

        hy_path = numpy.array(hy_path)
        hz_path = numpy.array(hz_path)

        #Gaussian with StDev of data
        g = numpy.exp( -numpy.power(hy_center-self.zHeights.mean(),2)/2/numpy.power(self.stdev_profile_heights(),2) )
        g = g/g.sum()*hz.sum()

        g_path = numpy.exp( -numpy.power(hy_path-self.zHeights.mean(),2)/2/numpy.power(self.stdev_profile_heights(),2) )
        g_path = g_path/g_path.sum()*hz_path.sum()

        self.histoHeights = {"x":hy_center, "y1":hz, "y2":g, "x_path":hy_path, "y1_path":hz_path, "y2_path":g_path}


    def _write_output_files(self):

        # write header file
        outFile = self.get_input_value("outputFileRoot") + "Header.txt"
        with open(outFile, mode='w') as f1:
            json.dump(self.metadata, f1, indent=2)
        if not(self.get_input_value("silent")):
            print ("File "+outFile+" containing metadata written to disk.")

        #
        # Dump heights and slopes profiles to files
        #
        outFile = self.get_input_value("outputFileRoot")+'Heights.dat'
        dd=numpy.concatenate( (self.y.reshape(-1,1), self.zHeights.reshape(-1,1)),axis=1)
        numpy.savetxt(outFile,dd,comments="#",header="F %s\nS 1  heights profile\nN 2\nL  coordinate[m]  height[m]"%(outFile))
        if not(self.get_input_value("silent")):
            print ("File "+outFile+" containing heights profile written to disk.")

        outFile = self.get_input_value("outputFileRoot")+'Slopes.dat'
        dd=numpy.concatenate( (self.y.reshape(-1,1), self.zSlopes.reshape(-1,1)),axis=1)
        numpy.savetxt(outFile,dd,comments="#",header="F %s\nS 1  slopes profile\nN 2\nL  coordinate[m]  slopes[rad]"%(outFile))
        if not(self.get_input_value("silent")):
            print ("File "+outFile+" written to disk.")


        #write psd file
        dd = numpy.concatenate( (self.f, self.psdHeights, self.psdSlopes, \
                                 numpy.sqrt(self.csdHeights)/self.stdev_psd_heights(), \
                                 numpy.sqrt(self.csdSlopes)/self.stdev_psd_slopes() \
                                 ) ,axis=0).reshape(5,-1).transpose()
        outFile = self.get_input_value("outputFileRoot")+'PSD.dat'
        header = "F %s\nS 1  power spectral density\nN 5\nL  freq[m^-1]  psd_heights[m^3]  psd_slopes[rad^3]  csd_h  csd_s"%(outFile)
        numpy.savetxt(outFile,dd,comments="#",header=header)
        if not(self.get_input_value("silent")):
            print ("File "+outFile+" written to disk.")


        # write slopes histogram
        dd=numpy.concatenate( (self.histoSlopes["x"],self.histoSlopes["y1"],self.histoSlopes["y2"] ) ,axis=0).reshape(3,-1).transpose()
        outFile = self.get_input_value("outputFileRoot")+'HistoSlopes.dat'
        numpy.savetxt(outFile,dd,header="F %s\nS  1  histograms of slopes\nN 3\nL  slope[rad] at bin center  counts  Gaussian with StDev = %g"%
                                        (outFile,self.stdev_profile_slopes()),comments='#')
        if not(self.get_input_value("silent")):
            print ("File "+outFile+" written to disk.")

        # heights histogram
        dd=numpy.concatenate( (self.histoHeights["x"],self.histoHeights["y1"],self.histoHeights["y2"] ) ,axis=0).reshape(3,-1).transpose()
        outFile = self.get_input_value("outputFileRoot")+'HistoHeights.dat'
        numpy.savetxt(outFile,dd,header="F %s\nS  1  histograms of heights\nN 3\nL  heights[m] at bin center  counts  Gaussian with StDev = %g"%
                                        (outFile,self.stdev_profile_heights()),comments='#')

        # profiles info
        outFile = self.get_input_value("outputFileRoot")+'Info.txt'
        f = open(outFile,'w')
        f.write(self.info_profiles())
        f.close()


    def _write_file_for_shadow(self):
        #
        #  write file for SHADOW (optional)
        #  replicate the (x,z) profile in a "s" mesh of npointsx * npointsy
        #

        #inputs
        npointsy = int(self.get_input_value("shadowNy"))
        npointsx = int(self.get_input_value("shadowNx"))
        mirror_width = float(self.get_input_value("shadowWidth")) # in cm

        # units to cm
        y = (self.y).copy() * 100.0 # from m to cm
        z = (self.zHeights).copy() * 100.0 # from m to cm

        # set origin at the center of the mirror. TODO: allow any point for origin
        z = z - z.min()
        y = y - y[int(y.size/2)]


        # interpolate the profile (y,z) to have npointsy points (new profile yy,zz)
        if npointsy > 0:
            mirror_length = y.max() - y.min()
            yy = numpy.linspace(-mirror_length/2.0,mirror_length/2.0,npointsy)
            zz = numpy.interp(yy,y,z)

            # dump to file interpolated profile (for fun)
            if self.get_input_value("outputFileRoot") != "":
                dd = numpy.concatenate( (yy.reshape(-1,1), zz.reshape(-1,1)),axis=1)
                outFile = self.get_input_value("outputFileRoot") + "ProfileInterpolatedForShadow.dat"
                numpy.savetxt(outFile,dd)
                if not(self.get_input_value("silent")):
                    print("File %s with interpolated heights profile for SHADOW written to disk."%outFile)
        else:
            yy = y
            zz = z
            npointsy = yy.size

        # fill the mesh arrays xx,yy,s with interpolated profile yy,zz
        xx=numpy.linspace(-mirror_width/2.0,mirror_width/2.0,npointsx)
        s = numpy.zeros( (npointsy,npointsx) )
        for i in range(npointsx):
            s[:,i]=zz

        # write Shadow file
        outFile = self.get_input_value("outputFileRoot") + "Shadow.dat"
        tmp = write_shadowSurface(s,xx,yy,outFile=outFile)


    def _latex_line(self,table_number=1):
        """
        to create a line with profile data latex-formatted for automatic compilation of tables in the paper
        :return:
        """
        if table_number == 1:
            return  ('%d & %s & %d & %.2f  (%.2f %s) & %.2f  (%.2f %s) \\\\'%(   \
                self.get_input_value("entryNumber"),   \
                self.metadata['SURFACE_SHAPE'],
                int(1e3*(self.y[-1]-self.y[0])),   \
                1e6*self.zSlopes.std(ddof=1),      \
                1e6*self.stdev_psd_slopes(),           \
                ("" if self.metadata['CALC_SLOPE_RMS'] is None else ",%.2f"%(self.metadata['CALC_SLOPE_RMS'])),    \
                1e9*self.stdev_psd_heights(),           \
                1e9*self.zHeights.std(ddof=1),   \
                ("" if self.metadata['CALC_HEIGHT_RMS'] is None else ",%.2f"%(self.metadata['CALC_HEIGHT_RMS'])),  ))
        else:
            return  ('%d & %d & %.2f & %.2f & %d & %.2f & %.2f & %.2f\\\\'%(   \
                self.get_input_value("entryNumber"),   \
                self.y.size, \
                self.momentsHeights[2], \
                self.momentsHeights[3],\
                ((autocorrelationfunction(self.y,self.zHeights))[2])*1e3, \
                -self.powerlaw["hgt_pendent"], \
                self.momentsSlopes[2], \
                self.momentsSlopes[3],\
                ))

    def _text_line(self):
        """
        to create a line with profile data ascii-formatted for automatic compilation of profile summary
        :return:
        """
        return  ('%3d  %12s %8.2f  %.2f %s %.2f %s'%(   \
            self.get_input_value("entryNumber"),   \
            self.metadata['SURFACE_SHAPE'],
            int(1e3*(self.y[-1]-self.y[0])),   \
            1e6*self.zSlopes.std(ddof=1),      \
            ("       " if self.metadata['CALC_SLOPE_RMS'] is None else "(%5.2f)"%(self.metadata['CALC_SLOPE_RMS'])),    \
            1e9*self.zHeights.std(ddof=1),   \
            ("       " if self.metadata['CALC_HEIGHT_RMS'] is None else "(%5.2f)"%(self.metadata['CALC_HEIGHT_RMS'])),  ))

    def _dictionary_line(self):
        """
        to create a dictionary with profile data for automatic compilation of profile summary
        :return:
        """
        return  {  \
            "entry":self.get_input_value("entryNumber"),   \
            "surface":self.metadata['SURFACE_SHAPE'],
            "length":(self.y[-1]-self.y[0]),   \
            "slp_err":self.zSlopes.std(ddof=1),      \
            "slp_err_user":self.metadata['CALC_SLOPE_RMS'], \
            "hgt_err":self.zHeights.std(ddof=1), \
            "hgt_err_user": self.metadata['CALC_HEIGHT_RMS'] }


#
# main functions (these function are sitting here for autoconsistency of dabam.py, otherwise can be in a dependency)
#

def cdf(sy, sz, method = 1 ):
    """
     cdf: Calculates the profile from the slope by simple integration (antiderivative)

      INPUTS:
           sy - 1D array of (equally-spaced) lengths.
           sz - 1D array of slopes.
      KEYWORDS
           method : 0 use simple sum as integration method
                    1 use trapezoidal rule (default)
      RESTRICTIONS:
          the abscissas step must be sorted, but may not be constant

           1D array with cdf

    """

    zprof = sz*0.0
    if method == 0:
        steps = sy[0:sz.size-1]
        steps = numpy.concatenate(([0],steps))
        steps[0] = steps[1]
        steps.shape = -1
        steps = sy - steps
        zprof = numpy.cumsum(sz*steps)
    else:
        for i in range(sz.size):
          zprof[i]= numpy.trapz(sz[0:i+1], x = sy[0:i+1])

    return zprof


def psd(xx, yy, onlyrange = None):
    """
     psd: Calculates the PSD (power spectral density) from a profile

      INPUTS:
           x - 1D array of (equally-spaced) lengths.
           y - 1D array of heights.
      OUTPUTS:
           f - 1D array of spatial frequencies, in units of 1/[x].
           s - 1D array of PSD values, in units of [y]^3.
      KEYWORD PARAMETERS:
           onlyrange - 2-element array specifying the min and max spatial
               frequencies to be considered. Default is from
               1/(length) to 1/(2*interval) (i.e., the Nyquist
               frequency), where length is the length of the scan,
               and interval is the spacing between points.

      PROCEDURE
            Use FFT

    """
    n_pts = xx.size
    if (n_pts <= 1):
        print ("psd: Error, must have at least 2 points.")
        return 0

    window=yy*0+1.
    length=xx.max()-xx.min()  # total scan length.
    delta = xx[1] - xx[0]

    # psd from windt code
    # s=length*numpy.absolute(numpy.fft.ifft(yy*window)**2)
    # s=s[0:(n_pts/2+1*numpy.mod(n_pts,2))]  # take an odd number of points.

    #xianbo + luca:
    s0 = numpy.absolute(numpy.fft.fft(yy*window))
    s =  2 * delta * s0[0:int(len(s0)/2)]**2/s0.size # uniformed with IGOR, FFT is not symmetric around 0
    s[0] /= 2
    s[-1] /= 2


    n_ps=s.size                       # number of psd points.
    interval=length/(n_pts-1)         # sampling interval.
    f_min=1./length                   # minimum spatial frequency.
    f_max=1./(2.*interval)            # maximum (Nyquist) spatial frequency.
    # spatial frequencies.
    f=numpy.arange(float(n_ps))/(n_ps-1)*(f_max-f_min)+f_min

    if onlyrange != None:
        roi =  (f <= onlyrange[1]) * (f >= onlyrange[0])
        if roi.sum() > 0:
            roi = roi.nonzero()
            f = f[roi]
            s = s[roi]

    return s,f


def autocorrelationfunction(x,f):
    """
    calculates the autocovariance function and correlation length of a 1-d profile f(x).

    Adapted from matlab code acf1D (David Bergstrom)
    see http://www.mysimlabs.com/surface_generation.html

    :param x: the abscissas array (profile points)
    :param f: array with the funcion value (profile heights)
    :return: lags,acf,cl
             lags - lag length vector (abscissas of acf)
             acf  - autocovariance function
             cl   - correlation length
    """

    # function [acf,cl,lags] = acf1D(f,x,opt)
    # %
    # % [acf,cl,lags] = acf1D(f,x)
    # %
    # % calculates the autocovariance function and correlation length of a
    # % 1-d profile f(x).
    # %
    # % Input:    x    - profile points
    # %           f    - profile heights
    # %
    # % Output:   lags - lag length vector (useful for plotting the acf)
    # %           acf  - autocovariance function
    # %           cl   - correlation length
    # %
    # % Last updated: 2010-07-26 (David Bergstrom)
    # %

    N = len(x)
    lags = numpy.linspace(0,x[-1]-x[0],N)
    # c=xcov(f,'coeff'); % the autocovariance function
    f -= f.mean()
    c = numpy.convolve(f,f[::-1])
    c = c / c.max()
    acf=c[(N-1):2*N-2]
    k = 0

    while acf[k] > 1/numpy.exp(1):
        k = k + 1

    cl = 1/2*(x[k-1]+x[k]-2*x[0])

    return lags,acf,cl


#
def func_ellipse_slopes(x, p, q, theta, shift):
    """
    calculates the derivative (y'(x) i.e., slopes) of a ellipse y(x) defined by its distance to focii (p,q) and grazing
    angle theta at coordinate x=0
    :param x: the length coordinate for the ellipse (x=0 is the center)
    :param p: the distance from source to mirror center
    :param q: the distance from mirror center to image
    :param theta: the grazing incidence angle in rad
    :param shift: a vertical shift to be added to the ellipse y' coordinate.
    :return:
    """


    a = (p + q) / 2

    b = numpy.sqrt( numpy.abs(p * q)) * numpy.sin(theta)

    c = numpy.sqrt(numpy.abs(a*a - b*b))

    epsilon = c / a

    # (x0,y0) are the coordinates of the center of the mirror
    # x0 = (p*p - q*q) / 4 / c
    x0 = (p - q) / 2 / epsilon
    y0 = -b * numpy.sqrt(numpy.abs(1.0 - ((x0/a)**2)))

    # the versor normal to the surface at the mirror center is -grad(ellipse)
    xnor = -2 * x0 / a**2
    ynor = -2 * y0 / b**2
    modnor = numpy.sqrt(xnor**2 + ynor**2)
    xnor /= modnor
    ynor /= modnor
    # tangent  versor is perpendicular to normal versor
    xtan =  ynor
    ytan = -xnor

    A = 1/b**2
    B = 1/a**2
    C = A

    CCC = numpy.zeros(11) # this is for the general 3D case (we need 10 coeffs, index=0 not used here)
    # The 2D implicit ellipse equation is c2 x^2 + c3 y^2 + c5 x y + c8 x + c9 y + c10 = 0
    #CCC[1] = A
    CCC[2] = B*xtan**2 + C*ytan**2
    CCC[3] = B*xnor**2 + C*ynor**2
    #CCC[4] = .0
    CCC[5] = 2*(B*xnor*xtan+C*ynor*ytan)
    #CCC[6] = .0
    #CCC[7] = .0
    CCC[8] = .0
    CCC[9] = 2*(B*x0*xnor+C*y0*ynor)
    CCC[10]= .0

    #reorder in y and get the second degree equation for heights
    # AA y^2 + BB y + CC = 0
    AA = CCC[3]
    BB = CCC[5]*x + CCC[9]
    CC = CCC[2]*x*x + CCC[8]*x + CCC[10]
    DD = BB*BB-4*AA*CC

    #calculate derivatives and solve fir y' (P=primes)
    BBP = CCC[5]
    CCP = 2*CCC[2]*x+CCC[8]
    DDP = 2*BB*BBP -4*AA*CCP
    ells = (-1/2/AA) * (BBP + DDP/2/numpy.sqrt(DD))

    return ells+shift


def write_shadowSurface(s,xx,yy,outFile='presurface.dat'):
    """
      write_shadowSurface: writes a mesh in the SHADOW/presurface format

      SYNTAX: 
           out = write_shadowSurface(z,x,y,outFile=outFile)

      INPUTS:
           z - 2D array of heights
           x - 1D array of spatial coordinates along mirror width.
           y - 1D array of spatial coordinates along mirror length.
     
      OUTPUTS:
           out - 1=Success, 0=Failure
           outFile - output file in SHADOW format. If undefined, the
                     file is names "presurface.dat"
     
    """
    out = 1

    try:
       fs = open(outFile, 'w')
    except IOError:
       out = 0
       print ("Error: can\'t open file: "+outFile)
       return 
    else:
        # dimensions
        fs.write( "%d  %d \n"%(xx.size,yy.size))
        # y array
        for i in range(yy.size): 
            fs.write(' ' + repr(yy[i]) )
        fs.write("\n")
        # for each x element, the x value and the corresponding z(y) profile
        for i in range(xx.size): 
            tmps = ""
            for j in range(yy.size): 
                tmps = tmps + "  " + repr(s[j,i])
            fs.write(' ' + repr(xx[i]) + " " + tmps )
            fs.write("\n")
        fs.close()
        #print ("File "+outFile+" written to disk (for SHADOW).")

def moment(array,substract_one_in_variance_n=True):
    """
    Calculate the first four statistical moments of a 1D array
    :param array:
    :param substract_one_in_variance_n:
    :return: array with: m0 (mean) m1 (variance) m2 (skewness) m3 (kurtosis)
    """
    a1 = numpy.array(array)
    m0 = a1.mean()

    tmp = (a1-m0)**2
    if substract_one_in_variance_n:
        m1 = tmp.sum()/(a1.size-1)
    else:
        m1 = tmp.sum()/(a1.size)
    sd = numpy.sqrt(m1)

    tmp = (a1-m0)**3
    m2 = tmp.sum()/sd**3/a1.size

    tmp = (a1-m0)**4
    m3 = (tmp.sum()/sd**4/a1.size) - 3 #Fisher definition: substract 3 to return 0 for Normal distribution
    return m0,m1,m2,m3


def dabam_summary(nmax=None,latex=0):
    """
    create a text with the summary of all dabam entries

    :param nmax:
    :param latex:
    :return:
    """

    if nmax is None:
        nmax = 1000000  # this is like infinity
    if latex ==0:
        txt = "Entry    shape  Length[mm]  slp_err [urad]  hgt_err [um]\n"
    else:
        txt = ""


    for i in range(nmax):
        dm = dabam()
        dm.set_input_outputFileRoot("")  # avoid output files
        dm.set_input_silent(1)
        dm.set_entry(i+1)
        try:
            dm.load()
        except:
            break
        if latex == 1:
            txt += dm._latex_line(table_number=1)+"\n"
        elif latex == 2:
            txt += dm._latex_line(table_number=2)+"\n"
        else:
            txt += dm._text_line()+"\n"
    return(txt)
'''
def dabam_summary_dictionary():
    nmax = 1000000  # this is like infinity
    out = []
    for i in range(nmax):
        dm = dabam()
        dm.set_input_outputFileRoot("")  # avoid output files
        dm.set_input_silent(1)
        dm.set_entry(i+1)
        try:
            dm.load()
        except:
            break
        tmp = dm._dictionary_line()

        out.append(tmp)
    return(out)
'''

def dabam_summary_dictionary(surface=None, slp_err_from=None, slp_err_to=None, length_from=None, length_to=None):
    nmax = 1000000  # this is like infinity
    out = []
    for i in range(nmax):
        dm = dabam()
        dm.set_input_outputFileRoot("")  # avoid output files
        dm.set_input_silent(1)
        dm.set_entry(i+1)
        try:
            dm.load()
        except:
            break
        tmp = dm._dictionary_line()

        add_element = True
        if not surface is None: add_element = tmp["surface"] == surface
        if add_element and not slp_err_from is None and not slp_err_to is None: add_element = tmp["slp_err"] >= slp_err_from and tmp["slp_err"] <= slp_err_to
        if add_element and not length_from is None and not length_to is None: add_element = tmp["length"] >= length_from and tmp["length"] <= length_to
        if add_element: out.append(tmp)
    return(out)


#
# tests
#
def test_dabam_names():
    """
    Tests that the I/O methods work well for the list of input values
    :return:
    """

    print("-------------------  test_dabam_names ------------------------------")
    dm = dabam()
    number_of_input_fields = len(dm.inputs)

    argsdict = dm.inputs
    names = []
    values = []
    for i,j in argsdict.items():
        names.append(i)
        values.append(j)

    #change values and reinsert in object
    values2 = copy.copy(values)
    for i in range(number_of_input_fields):
        if values[i] != None:
            values2[i] = 2*values[i]

    print ("-----------------------------------------------------")
    print ("--input_name value value2")
    for i in range(number_of_input_fields):
        print(i,names[i],values[i],values2[i])
        dm.inputs[names[i]] = values2[i]
    print ("-----------------------------------------------------")


    print ("-----------------------------------------------------")
    print ("--input_name input_name_short stored_value2, help")
    for i in range(number_of_input_fields):

        print(names[i],
            dm.get_input_value(names[i]),
            dm.get_input_value_short_name(names[i]),
            dm.inputs[names[i]],"\n",
            dm.get_input_value_help(names[i]),
              )
    print ("-----------------------------------------------------")


    #back to initial values
    dict2 = dm.get_inputs_as_dictionary()
    for i in range(number_of_input_fields):
        dict2[names[i]] = values[i]
        dm.inputs[names[i]] = values2[i]
    dm.set_inputs_from_dictionary(dict2)

    print ("--back to initial value")
    if (dm.inputs == dabam().inputs):
        print("Back to initial value: OK")
    else:
        raise Exception("Back to initial value: error returning to initial state")

def test_dabam_stdev_slopes():
    """
    Tests the slope error value for the nmax first profiles (from remote server)
    :return:
    """

    print("-------------------  test_dabam_slopes ------------------------------")
    stdev_ok =  [4.8651846141972904e-07, 1.5096270252538352e-07, 1.7394444580303415e-07, 1.3427931903345248e-07, 8.4197811681221573e-07, 1.0097219914737401e-06, 5.74153915948042e-07, 5.7147678897188605e-07, 4.3527688789008779e-07, 2.3241765005153794e-07, 2.2883095949050537e-07, 3.1848792295534762e-07, 1.2899449478710491e-06, 1.1432193606225235e-06, 2.1297554130432642e-06, 1.8447156600570902e-06, 2.2715775271373941e-06, 1.1878208663183125e-07, 4.1777346923623561e-08, 4.0304426129060434e-07, 4.3430016136041185e-07, 5.3156037926371151e-06, 1.7725086287871762e-07, 2.0222947541222619e-07, 7.2140041229621698e-08]

    nmax = 25

    tmp_profile = []
    tmp_psd = []
    for i in range(nmax):
        print(">> testing slopes stdev from profile number: ",i )
        dm = dabam()
        dm.set_input_silent(True)
        dm.set_input_entryNumber(i+1)
        dm.load()
        stdev_profile = dm.stdev_profile_slopes()
        stdev_psd = dm.stdev_psd_slopes()
        tmp_profile.append(stdev_profile)
        tmp_psd.append(stdev_psd)

    print("stdev from profile:          ",repr(tmp_profile))
    print("stdev from psd:              ",repr(tmp_psd))
    print("stdev OK (stored):           ",repr(stdev_ok))

    for i in range(nmax):
        print("Checking correctness of dabam-entry: %d"%(1+i))
        print("    Check slopes profile urad:  StDev=%f, FromPSD=%f, stored=%f "%(1e6*tmp_profile[i],1e6*tmp_psd[i],1e6*stdev_ok[i]))
        assert abs(tmp_profile[i] - stdev_ok[i])<1e-9
        assert abs(tmp_psd[i] - stdev_ok[i])<1e-8


#
# main program
#
def main():

    # initialize
    dm = dabam()

    dm.set_input_outputFileRoot("tmp") # write files by default
    dm._set_from_command_line()   # get arguments of dabam command line


    if dm.get_input_value("runTests"): # if runTests selected
        dm.set_input_outputFileRoot("")      # avoid output files
        test_dabam_names()
        test_dabam_stdev_slopes()
    elif dm.get_input_value("summary"):
        print(dabam_summary())
    else:
        dm.load()        # access data

        if dm.get_input_value("plot") != None:
            dm.plot()

#
# main call
#
if __name__ == '__main__':
    main()

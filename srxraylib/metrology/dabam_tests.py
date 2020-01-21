#
# tests
#

from dabam import dabam
import copy
from numpy.testing import assert_almost_equal

def data1():  # some scratch data
    return """Line1	Line2	Center	Line3	Line4
21.1	-0.871333347	-1.536403196	-1.826784168	-2.013347664	-2.058588716
21.15	-0.876013666	-1.51561079	-1.868885951	-2.014787719	-1.968877289
21.2	-0.981523985	-1.708348394	-1.928961741	-1.979515764	-2.158752833
21.25	-1.087034305	-1.901102009	-1.989037538	-1.944227798	-2.348627347
21.3	-1.254296626	-1.936110635	-2.020671342	-1.791950822	-2.355610831
21.35	-1.421559948	-1.971134271	-2.052305152	-1.639659835	-2.362594286
21.4	-1.399064269	-1.861595918	-2.10292097	-1.713075837	-2.203912711
21.45	-1.376568592	-1.752073576	-2.153552794	-1.78649183	-2.045231106
21.5	-1.036460915	-1.787997245	-2.159108625	-1.864530811	-2.085005472
21.55	-0.696369238	-1.823905924	-2.164680463	-1.942569782	-2.124763808
21.6	-0.780425563	-1.778026614	-2.098917308	-1.897776743	-2.114199114
21.65	-0.864466887	-1.732164315	-2.03315516	-1.852982693	-2.103650391
21.7	-0.940757213	-1.772101026	-2.044830018	-1.900870632	-2.265402638
21.75	-1.017046539	-1.812052748	-2.056520884	-1.948758561	-2.427155856
21.8	-0.989683865	-1.840515481	-2.067112757	-2.05128848	-2.448787044
21.85	-0.962320192	-1.868978225	-2.077720636	-2.153833388	-2.470418202
21.9	-1.01848352	-1.887124979	-1.841470522	-2.060653285	-2.310500331
21.95	-1.074646848	-1.905287744	-1.605221415	-1.967489172	-2.15056743
22	-1.200116177	-1.95608952	-1.666137315	-1.785961048	-2.117633499
22.05	-1.325584506	-2.006875306	-1.727037222	-1.604417914	-2.084699539
22.1	-1.108096836	-1.908324104	-1.757405136	-1.56421777	-2.053077549
22.15	-0.890594166	-1.809756912	-1.787772056	-1.524000614	-2.021439529
22.2	-0.942897497	-1.81974073	-1.763009984	-1.757771449	-2.05019648
22.25	-0.995215829	-1.82972456	-1.738246918	-1.991542273	-2.078953401
22.3	-1.071337161	-1.8554104	-1.976499859	-2.095033086	-2.194563292
22.35	-1.147459494	-1.88108025	-2.214768808	-2.198523889	-2.310189154
22.4	-0.956460827	-1.821011112	-2.387439763	-2.240735681	-2.213289986
22.45	-0.765463161	-1.760941984	-2.560125725	-2.282946463	-2.116405789
"""

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

def test_dabam_stdev_slopes(nmax=9):
    """
    Tests the slope error value for the nmax first profiles (from remote server)
    :return:
    """

    print("-------------------  test_dabam_slopes ------------------------------")
    stdev_ok =  [4.8651846141972904e-07, 1.5096270252538352e-07, 1.7394444580303415e-07, 1.3427931903345248e-07, 8.4197811681221573e-07, 1.0097219914737401e-06, 5.74153915948042e-07, 5.7147678897188605e-07, 4.3527688789008779e-07, 2.3241765005153794e-07, 2.2883095949050537e-07, 3.1848792295534762e-07, 1.2899449478710491e-06, 1.1432193606225235e-06, 2.1297554130432642e-06, 1.8447156600570902e-06, 2.2715775271373941e-06, 1.1878208663183125e-07, 4.1777346923623561e-08, 4.0304426129060434e-07, 4.3430016136041185e-07, 5.3156037926371151e-06, 1.7725086287871762e-07, 2.0222947541222619e-07, 7.2140041229621698e-08]


    tmp_profile = []
    tmp_psd = []
    stdev_ok = []
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
        try:
            tmp = float(dm.metadata["CALC_SLOPE_RMS"]) * float(dm.metadata["CALC_SLOPE_RMS_FACTOR"])
        except:
            tmp = 0
        stdev_ok.append(tmp)

    for i in range(nmax):
        print("Entry, stdev from profile,  stdev from psd, stdev OK (stored): %03d  %8.3g  %8.3g  %8.3g"%
              (i+1,tmp_profile[i],tmp_psd[i],stdev_ok[i]))

    for i in range(nmax):
        if stdev_ok[i] != 0.0:
            print("Checking correctness of dabam-entry: %d"%(1+i))
            print("    Check slopes profile urad:  StDev=%f, FromPSD=%f, stored=%f "%(1e6*tmp_profile[i],1e6*tmp_psd[i],1e6*stdev_ok[i]))
            assert abs(tmp_profile[i] - stdev_ok[i])<1e-6
            assert abs(tmp_psd[i] - stdev_ok[i])<1e-6


def test_entry():
    dm = dabam.initialize_from_entry_number(80)


    stdev_profile = dm.stdev_profile_slopes()
    stdev_psd = dm.stdev_psd_slopes()

    assert_almost_equal(stdev_profile, stdev_psd)
    assert_almost_equal(stdev_profile, 0.158e-6)
    assert_almost_equal(stdev_psd, 0.158e-6)

def test_entry_elliptical():
    dm = dabam.initialize_from_entry_number(4)

    # recalculate without detrending
    dm.set_input_setDetrending(-1)
    dm.make_calculations()


def test_entry_text():

    txt = data1().split("\n")
    dm = dabam.initialize_from_external_data(txt,
                               column_index_abscissas=0,
                               column_index_ordinates=1,
                               skiprows=1,
                               useHeightsOrSlopes=0,
                               to_SI_abscissas=1e-3,
                               to_SI_ordinates=1e-9,
                               detrending_flag=-1)

def test_entry_file():
    filename = "tmp.dat"
    f = open(filename,"w")
    f.write(data1())
    f.close()
    print("File written to disk: %s"%filename)

    dm = dabam.initialize_from_external_data(filename,
                               column_index_abscissas=0,
                               column_index_ordinates=1,
                               skiprows=1,
                               useHeightsOrSlopes=0,
                               to_SI_abscissas=1e-3,
                               to_SI_ordinates=1e-9,
                               detrending_flag=-1)


def test_write_dabam_formatted_files():

    txt = data1().split("\n")
    dm = dabam.initialize_from_external_data(txt,
                               column_index_abscissas=0,
                               column_index_ordinates=1,
                               skiprows=1,
                               useHeightsOrSlopes=0,
                               to_SI_abscissas=1e-3,
                               to_SI_ordinates=1e-9,
                               detrending_flag=-1)

    dm.write_output_dabam_files(filename_root="tmp-DABAM-XXX",loaded_from_file=txt)

    dm.metadata_set_info(YEAR_FABRICATION=2019) # fill metadata info
    dm.write_output_dabam_files(filename_root="tmp-DABAM-YYY")


def test_local_server():

    import urllib.request
    urllib.request.urlretrieve ("http://ftp.esrf.eu/pub/scisoft/dabam/data/dabam-081.txt", "/tmp/dabam-081.txt")
    urllib.request.urlretrieve("http://ftp.esrf.eu/pub/scisoft/dabam/data/dabam-081.dat", "/tmp/dabam-081.dat")
    dm = dabam.initialize_from_local_server(81,"/tmp")
    m0 = dm.momentsHeights, dm.momentsSlopes

    # reset to remote server

    dm.set_default_server()
    dm.set_input_entryNumber(81)
    dm.make_calculations()

    m1 = dm.momentsHeights, dm.momentsSlopes

    # set again to local serer
    dm = dabam.initialize_from_local_server(81,"/tmp")
    m2 = dm.momentsHeights, dm.momentsSlopes

    assert_almost_equal(m0, m1)
    assert_almost_equal(m0, m2)


def test_summary_dictionary():
    dm = dabam()
    out1 = dm.dabam_summary_dictionary_from_json_indexation(surface=None,
                                                            slp_err_from=None,
                                                            slp_err_to=None,
                                                            length_from=None,
                                                            length_to=None)

    out2 = dm.dabam_summary_dictionary_from_scratch(surface=None,
                                                            slp_err_from=None,
                                                            slp_err_to=None,
                                                            length_from=None,
                                                            length_to=None,
                                                            verbose=True)
    for i,ilist in enumerate(out1):
        for key in ilist.keys():
            # print(i,key,out1[i][key],out2[i][key])
            if isinstance(out1[i][key],float):
                assert_almost_equal(out1[i][key],out2[i][key])
            else:
                assert (out1[i][key] == out2[i][key])

def test_load_dictionary():
    from dabam import dabam_summary_dictionary
    d = dabam_summary_dictionary(surface=None,
                                 slp_err_from=None,
                                 slp_err_to=None,
                                 length_from=None,
                                 length_to=None,
                                 verbose=True,
                                 server=None)

    print(d)

if __name__ == "__main__":

    test_entry()
    test_entry_file()
    test_entry_elliptical()
    test_entry_text()
    test_dabam_names()
    test_dabam_stdev_slopes()

    test_write_dabam_formatted_files()
    test_write_dabam_formatted_files()

    test_local_server()
    test_load_dictionary()

    # test_summary_dictionary() #slow...


    # filename = "/tmp/tmp18"
    # f = open(filename,'r')
    # txt = f.readlines()
    #
    # dm = dabam.initialize_from_external_data(txt,
    #                                          column_index_abscissas=0,
    #                                          column_index_ordinates=4,
    #                                          skiprows=1,
    #                                          useHeightsOrSlopes=0,
    #                                          to_SI_abscissas=1e-3,
    #                                          to_SI_ordinates=1e-9,
    #                                          detrending_flag=-3)



# SR-xraylib

Miscellaneous utilities and tools for synchrotron radiation and x-ray optics.

The classes and functions here can be used in a standalone more. They are separated in different folders: 

- metrology: tools for x-ray optics metrology
  - dabam: [http://ftp.esrf.eu/pub/scisoft/dabam/readme.html]
- plot: plotting tools
  - gol (Graphics in One Line): some functions to make easily simple plots using matplotlib
- sources: 
  - srfunc: functions to calculate synchrotron emission of bending magnets and wigglers
- util: 
  - data_structors: igor-like data structures
  - h5_simple_writer: simple writer for hdf5 files
  - inverse_method_sampler: classes for generating random numbers following given 1D, 2D and 3D probability dostribution functopns
- waveoptics: generic wavefront and free space propagators for x-ray optics (testing code, the final user-code is in [https://github.com/oasys-kit/wofry])
 


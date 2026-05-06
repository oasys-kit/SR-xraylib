.. currentmodule:: srxraylib

===========
Package API
===========
This page lists main classes in this package.


``srxraylib`` classes.


metrology
---------
``srxraylib.metrology`` classes and functions for metrology

.. autosummary::
   :toctree: generated/

   metrology.dabam
   metrology.profiles_simulation
   metrology.error_profile_calculator
   metrology.make_json_summary

plot
----
``srxraylib.plot`` functions for plots

.. autosummary::
   :toctree: generated/

   plot.gol

sources
-------
``srxraylib.sources`` functions for synchrotron radiation sources

.. autosummary::
   :toctree: generated/

   sources.srfunc

profiles
--------
``srxraylib.profiles`` mirror profile calculators

* ``srxraylib.profiles.benders`` mirror bender managers

.. autosummary::
   :toctree: generated/

   profiles.benders.bender_io
   profiles.benders.bender_manager
   profiles.benders.fixed_rods_bender_manager
   profiles.benders.flexural_hinge_bender_manager

* ``srxraylib.profiles.diaboloid`` diaboloid mirror shape

.. autosummary::
   :toctree: generated/

   profiles.diaboloid.diaboloid_calculator
   profiles.diaboloid.fqs

util
----
``srxraylib.util`` classes and functions with utilities and tools

.. autosummary::
   :toctree: generated/

   util.data_structures
   util.h5_simple_writer
   util.inverse_method_sampler
   util.chemical_formula
   util.custom_distribution
   util.histograms
   util.random_distributions
   util.threading

waveoptics
----------
``srxraylib.waveoptics`` 1D and 2D wave-optics tools (deprecated, use wofry/wofrylib)

.. autosummary::
   :toctree: generated/

   waveoptics.wavefront
   waveoptics.wavefront2D
   waveoptics.propagator
   waveoptics.propagator2D
   waveoptics.polarization
   waveoptics.CompactAFReader
"""
FEA_File: read a deformed surface (finite element analysis output, or an OASYS surface file), interpolate it
on a regular grid, post-process it and write it as an OASYS surface file (hdf5).

Input formats (``file_in_type``)
--------------------------------
0. ALS (text):  columns  A X Y Z DX DY DZ  (A: node index, ignored)
1. ESRF (text, comma separated):  columns  X, Y, Z, DX, DY, DZ
2. OASYS surface file (hdf5, local file or http/ftp url): datasets surface_file/X, surface_file/Y, surface_file/Z

For the text formats X, Y, Z are the undeformed node coordinates and DX, DY, DZ the deformations [m]; the
deformed surface is (X+DX, Y+DY, Z+DZ). The coordinates can be scaled with ``factorX``, ``factorY``, ``factorZ``
(e.g. 1e-3 for files in mm).

Array conventions
-----------------
* Internal 2D arrays use ``Z_INTERPOLATED[ix, iy]``, shape ``(len(x_interpolated), len(y_interpolated))``
  (as in shadow4).
* OASYS hdf5 surface files (and OasysSurfaceData) store the transposed array, ``Z[iy, ix]``;
  :meth:`FEA_File.write_h5_surface` and the reader take care of the transposition.
* Axes are named as in ANSYS: axis 0 = X, axis 1 = Y. With ``invert_axes_names=True`` the output file swaps
  the X and Y names.

Typical use
-----------
>>> o = FEA_File("disp.csv", file_in_type=1, factorX=1e-3, factorY=1e-3, factorZ=1e-3, load=True)
>>> o.replicate_raw_data(0)                # optional mirroring (0=no, 1=axis 0, 2=axis 1, 3=both)
>>> o.triangulate()                        # needed for point clouds (skipped for OASYS hdf5 files)
>>> o.interpolate_grid(801, 101)           # centred grid of 801 x 101 points
>>> o.detrend_best_circle(axis=0, fitting_domain_ratio=0.5)   # optional post-processing
>>> o.reset_height_to_central_value()
>>> o.write_h5_surface("disp.h5")

or, in one call, :meth:`FEA_File.process_file`.

Dependencies: numpy and scipy; h5py for OASYS hdf5 files (imported when needed); matplotlib and srxraylib.plot
only for the plotting methods.
"""
import os
import time
import numpy

from scipy import interpolate
from scipy import spatial
from scipy.interpolate import RectBivariateSpline

# matplotlib (and srxraylib.plot) are imported only inside the plotting methods: reading, interpolating,
# post-processing and writing do not need them.

#
# OASYS surface file (hdf5) i/o. Same layout as oasys2.widget.util.widget_util.read/write_surface_file
# (srxraylib does not depend on oasys2): group "surface_file" with datasets X, Y and Z[iy, ix].
#
SURFACE_FILE_GROUP = "surface_file"


def read_surface_file(file_name):
    """
    Read an OASYS surface file.

    Returns
    -------
    tuple
        (X, Y, Z) with Z[iy, ix], shape (len(Y), len(X)).
    """
    import h5py
    if not os.path.isfile(file_name): raise ValueError("File " + file_name + " not existing")
    with h5py.File(file_name, "r") as file:
        xx = file[SURFACE_FILE_GROUP + "/X"][()]
        yy = file[SURFACE_FILE_GROUP + "/Y"][()]
        zz = file[SURFACE_FILE_GROUP + "/Z"][()]
    return xx, yy, zz


def write_surface_file(zz, xx, yy, file_name, overwrite=True):
    """
    Write an OASYS surface file.

    Parameters
    ----------
    zz : numpy.ndarray
        Heights Z[iy, ix], shape (len(yy), len(xx)).
    xx, yy : numpy.ndarray
        Coordinates [m].
    file_name : str
        Output file name.
    overwrite : bool, optional
        Remove an existing file first.
    """
    import h5py
    if os.path.isfile(file_name) and overwrite: os.remove(file_name)

    with h5py.File(file_name, "a") as file:
        if "default" not in file.attrs:
            file.attrs['default']      = SURFACE_FILE_GROUP
            file.attrs['file_name']    = file_name
            file.attrs['file_time']    = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            file.attrs['creator']      = 'srxraylib.profiles.FEA_File'
            file.attrs['code']         = 'Oasys'
            file.attrs['HDF5_Version'] = h5py.version.hdf5_version
            file.attrs['h5py_version'] = h5py.version.version

        f1 = file.require_group(SURFACE_FILE_GROUP)
        f1z = f1.create_dataset("Z", data=zz)
        f1x = f1.create_dataset("X", data=xx)
        f1y = f1.create_dataset("Y", data=yy)

        # NEXUS attributes for automatic plot
        f1.attrs['NX_class'] = 'NXdata'
        f1.attrs['signal']   = "Z"
        f1.attrs['axes']     = [b"Y", b"X"]

        f1z.attrs['interpretation'] = 'image'
        f1x.attrs['long_name'] = "X [m]"
        f1y.attrs['long_name'] = "Y [m]"


class FEA_File():
    """
    A deformed surface: raw data (point cloud or regular grid) and its interpolation on a regular grid.

    Parameters
    ----------
    filename : str, optional
        Input file name or url (url only for OASYS hdf5 files).
    file_in_type : int, optional
        Input format: 0=ALS cols A X Y Z DX DY DZ, 1=ESRF cols X,Y,Z,DX,DY,DZ, 2=OASYS surface file (hdf5).
    skiprows : int, optional
        Number of header lines to skip (text formats).
    factorX, factorY, factorZ : float, optional
        Scaling factors applied to the coordinates when reading (e.g. 1e-3 for mm -> m).
    load : bool, optional
        If True, read the file at creation (see :meth:`load_multicolumn_file`).

    Attributes
    ----------
    Xundeformed, Yundeformed, Zundeformed : numpy.ndarray
        Undeformed node coordinates [m] (1D arrays).
    Xdeformation, Ydeformation, Zdeformation : numpy.ndarray
        Node deformations [m] (1D arrays). The deformed surface is given by :meth:`get_deformed`.
    x_interpolated, y_interpolated : numpy.ndarray
        Coordinates of the regular grid [m] (1D arrays).
    Z_INTERPOLATED : numpy.ndarray
        Heights on the regular grid [m], ``Z_INTERPOLATED[ix, iy]``.
    tri : scipy.spatial.Delaunay or None
        Triangulation of the deformed (x, y) points (point clouds only).
    """

    def __init__(self, filename="", file_in_type=0, skiprows=0, factorX=1.0, factorY=1.0, factorZ=1.0, load=False):
        self.filename = filename
        self.file_in_type = file_in_type
        self.skiprows = skiprows
        self.factorX = factorX
        self.factorY = factorY
        self.factorZ = factorZ
        self.reset()
        if load:
            self.load_multicolumn_file()

    def reset(self):
        """Clear the data (raw points, triangulation and interpolated grid); the file settings are kept."""
        self.Xundeformed = None  # 1D array
        self.Yundeformed = None  # 1D array
        self.Zundeformed = None  # 1D array

        self.Xdeformation = None  # 1D array
        self.Ydeformation = None  # 1D array
        self.Zdeformation = None  # 1D array

        self.triPi = None
        self.tri = None

        self.x_interpolated = None  # 1D array
        self.y_interpolated = None  # 1D array
        self.Z_INTERPOLATED = None  # 2D array

    @classmethod
    def process_file(cls, filename_in, n_axis_0=301, n_axis_1=51,
                     filename_out="", invert_axes_names=False,
                     detrend=0,
                     reset_height_method=0,
                     replicate_raw_data_flag=0,
                     file_in_type=0, skiprows=0,
                     do_plot=False):
        """
        Read, interpolate, post-process and (optionally) write a surface in one call.

        Parameters
        ----------
        filename_in : str
            Input file (see the class parameters for the formats).
        n_axis_0, n_axis_1 : int, optional
            Number of points of the interpolated grid (for OASYS hdf5 files, <= 0 means no interpolation).
        filename_out : str, optional
            If not empty, OASYS hdf5 file to write.
        invert_axes_names : bool, optional
            Swap the X and Y names in the output file.
        detrend : int, optional
            0=none, 1 (2)=straight line along axis 0 (1), 3 (4)=best circle along axis 0 (1).
        reset_height_method : int, optional
            0=none, 1=minimum height set to 0, 2=height at the centre set to 0.
        replicate_raw_data_flag : int, optional
            0=none, 1=mirror along axis 0, 2=along axis 1, 3=along both axes.
        file_in_type : int, optional
            Input format (0=ALS, 1=ESRF, 2=OASYS hdf5).
        skiprows : int, optional
            Header lines to skip (text formats).
        do_plot : bool, optional
            Show intermediate plots (matplotlib).

        Returns
        -------
        FEA_File
        """
        o1 = FEA_File(filename=filename_in, file_in_type=file_in_type, skiprows=skiprows, load=True)

        o1.replicate_raw_data(replicate_raw_data_flag)

        o1.triangulate()

        if do_plot:
            o1.plot_triangulation()

        o1.interpolate_grid(n_axis_0, n_axis_1)
        if do_plot:
            o1.plot_interpolated()

        if do_plot:
            o1.plot_surface_image()

        if detrend == 0:
            pass
        elif detrend == 1:
            o1.detrend_straight_line(axis=0)
        elif detrend == 2:
            o1.detrend_straight_line(axis=1)
        elif detrend == 3:
            o1.detrend_best_circle(axis=0)
        elif detrend == 4:
            o1.detrend_best_circle(axis=1)

        if reset_height_method == 0:
            pass
        elif reset_height_method == 1:
            o1.reset_height_to_minimum()
        elif reset_height_method == 2:
            o1.reset_height_to_central_value()

        if do_plot:
            o1.plot_surface_image()

        if filename_out != "":
            o1.write_h5_surface(filename=filename_out, invert_axes_names=invert_axes_names)

        return o1

    def set_filename(self, filename):
        """Set the input file name or url."""
        self.filename = filename

    def set_file_in_type(self, value):
        """Set the input format (0=ALS, 1=ESRF, 2=OASYS hdf5)."""
        self.file_in_type = value

    def load_multicolumn_file(self, skiprows=None, factorX=None, factorY=None, factorZ=None, file_in_type=None):
        """
        Read the input file (``self.filename``).

        Arguments left to None use the values given at creation (or set before); given values are stored.
        Text formats fill the node arrays; OASYS hdf5 files fill the regular grid directly (no interpolation
        needed) and the node arrays from it.

        Parameters
        ----------
        skiprows : int, optional
            Header lines to skip (text formats).
        factorX, factorY, factorZ : float, optional
            Scaling factors of the coordinates.
        file_in_type : int, optional
            0=ALS cols A X Y Z DX DY DZ, 1=ESRF cols X,Y,Z,DX,DY,DZ, 2=OASYS surface file (hdf5).
        """
        if skiprows is not None: self.skiprows = skiprows
        if factorX is not None: self.factorX = factorX
        if factorY is not None: self.factorY = factorY
        if factorZ is not None: self.factorZ = factorZ
        if file_in_type is not None: self.set_file_in_type(file_in_type)

        skiprows, factorX, factorY, factorZ = self.skiprows, self.factorX, self.factorY, self.factorZ
        file_in_type = self.file_in_type

        print("Reading file/url: %s" % self.filename)
        if file_in_type == 0: # ALS
            node = numpy.loadtxt(self.filename, skiprows=skiprows, dtype=numpy.float64)

            # Coordinates
            self.Xundeformed = factorX * node[:, 1]  # X=x in m
            self.Yundeformed = factorY * node[:, 2]  # Y=z in m
            self.Zundeformed = factorZ * node[:, 3]  # Z=uy vertical displacement in m

            self.Xdeformation = factorX * node[:, 4]  # X=x in m
            self.Ydeformation = factorY * node[:, 5]  # Y=z in m
            self.Zdeformation = factorZ * node[:, 6]  # Z=uy vertical displacement in m
        elif file_in_type == 1: # ESRF
            node = numpy.loadtxt(self.filename, skiprows=skiprows, dtype=numpy.float64, delimiter=',')

            # Coordinates
            self.Xundeformed = factorX * node[:, 0]  # X=x in m
            self.Yundeformed = factorY * node[:, 1]  # Y=z in m
            self.Zundeformed = factorZ * node[:, 2]  # Z=uy vertical displacement in m

            self.Xdeformation = factorX * node[:, 3]  # X=x in m
            self.Ydeformation = factorY * node[:, 4]  # Y=z in m
            self.Zdeformation = factorZ * node[:, 5]  # Z=uy vertical displacement in m

        elif file_in_type == 2: # HDF5 Oasys surface file

            downloaded = self.filename[0:4] == "http" or self.filename[0:3] == "ftp"
            if downloaded:
                import urllib.request
                filehandle, _ = urllib.request.urlretrieve(self.filename)
                print("Reading local file downloaded from url: %s" % filehandle)
            else:
                filehandle = self.filename

            try:
                self.surface_file_name = os.path.abspath(filehandle)
                xx, yy, zz = read_surface_file(filehandle)
            finally:
                if downloaded:  # remove the temporary copy
                    try: os.remove(filehandle)
                    except OSError: pass

            #
            # this file type does not need interpolation as it comes in a regular grid
            # (file Z[iy, ix] -> internal Z_INTERPOLATED[ix, iy])
            #
            self.x_interpolated = factorX * xx
            self.y_interpolated = factorY * yy
            self.Z_INTERPOLATED = factorZ * zz.copy().T

            self._set_points_from_grid()
        else:
            raise ValueError("file_in_type must be 0 (ALS), 1 (ESRF) or 2 (OASYS hdf5), got %r" % (file_in_type,))

    def _set_points_from_grid(self):
        # point arrays (used for limits, raw-data plots, ...) from the regular grid (OASYS surface file)
        XX = numpy.outer(self.x_interpolated, numpy.ones_like(self.y_interpolated))
        YY = numpy.outer(numpy.ones_like(self.x_interpolated), self.y_interpolated)

        self.Xdeformation = XX.flatten()  # X=x in m
        self.Ydeformation = YY.flatten()  # Y=z in m
        self.Zdeformation = self.Z_INTERPOLATED.flatten()  # Z=uy vertical displacement in m

        self.Xundeformed = numpy.zeros_like(self.Xdeformation)
        self.Yundeformed = numpy.zeros_like(self.Ydeformation)
        self.Zundeformed = numpy.zeros_like(self.Zdeformation)

    def Xdeformed(self):
        """Deformed X coordinates of the nodes [m]."""
        return self.Xundeformed + self.Xdeformation

    def Ydeformed(self):
        """Deformed Y coordinates of the nodes [m]."""
        return self.Yundeformed + self.Ydeformation

    def Zdeformed(self):
        """Deformed Z coordinates (heights) of the nodes [m]."""
        return self.Zundeformed + self.Zdeformation

    def get_deformed(self):
        """Returns (X, Y, Z) of the deformed nodes [m]."""
        return self.Xdeformed(),self.Ydeformed(),self.Zdeformed()

    def get_undeformed(self):
        """Returns (X, Y, Z) of the undeformed nodes [m]."""
        return self.Xundeformed,self.Yundeformed,self.Zundeformed

    def get_deformation(self):
        """Returns (DX, DY, DZ), the node deformations [m]."""
        return self.Xdeformation,self.Ydeformation,self.Zdeformation

    def get_limits_undeformed(self):
        """Returns (xmin, xmax, ymin, ymax, zmin, zmax) of the undeformed nodes."""
        return self.Xundeformed.min(), self.Xundeformed.max(), \
               self.Yundeformed.min(), self.Yundeformed.max(), \
               self.Zundeformed.min(), self.Zundeformed.max()

    def get_limits_deformation(self):
        """Returns (min, max) of DX, DY and DZ, as a 6-tuple."""
        return self.Xdeformation.min(), self.Xdeformation.max(), \
               self.Ydeformation.min(), self.Ydeformation.max(), \
               self.Zdeformation.min(), self.Zdeformation.max()

    def get_limits_deformed(self):
        """Returns (xmin, xmax, ymin, ymax, zmin, zmax) of the deformed nodes."""
        return self.Xdeformed().min(), self.Xdeformed().max(), \
               self.Ydeformed().min(), self.Ydeformed().max(), \
               self.Zdeformed().min(), self.Zdeformed().max()

    def get_dimensions(self):
        """Returns the number of nodes (as a 3-tuple, one value per coordinate array)."""
        return self.Xundeformed.size, self.Yundeformed.size, self.Zundeformed.size

    def replicate_raw_data(self,flag):
        """
        Mirror the data with respect to coordinate 0, keeping the heights (e.g. to build a full mirror from a
        half or a quarter model).

        Parameters
        ----------
        flag : int
            0=nothing, 1=mirror along axis 0 (x -> -x), 2=along axis 1 (y -> -y), 3=along both axes.

        Notes
        -----
        Point clouds: the nodes are duplicated. OASYS hdf5 files: the regular grid is mirrored (the line at
        coordinate 0 is not duplicated) and the data must lie on one side of 0 along each mirrored axis.
        """

        if flag == 0: # nothing
            return

        if self.file_in_type == 2: # OASYS surface file: mirror the regular grid
            if flag in (1, 3): self._mirror_grid(axis=0)
            if flag in (2, 3): self._mirror_grid(axis=1)
            self._set_points_from_grid()
            return

        if flag == 1: # axis 0
            self.Xundeformed = numpy.concatenate((-self.Xundeformed, self.Xundeformed))
            self.Yundeformed = numpy.concatenate((self.Yundeformed, self.Yundeformed))
            self.Zundeformed = numpy.concatenate((self.Zundeformed, self.Zundeformed))
            self.Xdeformation = numpy.concatenate((-self.Xdeformation, self.Xdeformation))
            self.Ydeformation = numpy.concatenate((self.Ydeformation, self.Ydeformation))
            self.Zdeformation = numpy.concatenate((self.Zdeformation, self.Zdeformation))
        elif flag == 2: # axis 1
            self.Xundeformed = numpy.concatenate((self.Xundeformed, self.Xundeformed))
            self.Yundeformed = numpy.concatenate((-self.Yundeformed, self.Yundeformed))
            self.Zundeformed = numpy.concatenate((self.Zundeformed, self.Zundeformed))
            self.Xdeformation = numpy.concatenate((self.Xdeformation, self.Xdeformation))
            self.Ydeformation = numpy.concatenate((-self.Ydeformation, self.Ydeformation))
            self.Zdeformation = numpy.concatenate((self.Zdeformation, self.Zdeformation))
        elif flag == 3: # both axes
            self.Xundeformed = numpy.concatenate((-self.Xundeformed, self.Xundeformed))
            self.Yundeformed = numpy.concatenate((self.Yundeformed, self.Yundeformed))
            self.Zundeformed = numpy.concatenate((self.Zundeformed, self.Zundeformed))
            self.Xdeformation = numpy.concatenate((-self.Xdeformation, self.Xdeformation))
            self.Ydeformation = numpy.concatenate((self.Ydeformation, self.Ydeformation))
            self.Zdeformation = numpy.concatenate((self.Zdeformation, self.Zdeformation))

            self.Xundeformed = numpy.concatenate((self.Xundeformed, self.Xundeformed))
            self.Yundeformed = numpy.concatenate((-self.Yundeformed, self.Yundeformed))
            self.Zundeformed = numpy.concatenate((self.Zundeformed, self.Zundeformed))
            self.Xdeformation = numpy.concatenate((self.Xdeformation, self.Xdeformation))
            self.Ydeformation = numpy.concatenate((-self.Ydeformation, self.Ydeformation))
            self.Zdeformation = numpy.concatenate((self.Zdeformation, self.Zdeformation))

    def _mirror_grid(self, axis=0):
        # Mirror the regular grid with respect to coordinate 0 along the given axis (keeping the heights),
        # as replicate_raw_data() does for point clouds. The data must lie on one side of 0 along that axis.
        coor = self.x_interpolated if axis == 0 else self.y_interpolated
        name = "axis 0 (X)" if axis == 0 else "axis 1 (Y)"
        if coor.min() < 0 < coor.max():
            raise Exception("Replicate along %s: the data must lie on one side of 0 (found [%g, %g])" %
                            (name, coor.min(), coor.max()))

        Z = self.Z_INTERPOLATED
        if coor[0] > coor[-1]: # make the coordinate increasing
            coor = coor[::-1]
            Z = numpy.flip(Z, axis=axis)

        if coor.max() <= 0:   # data on the negative side: mirror to the positive side
            keep = slice(None, -1) if coor[-1] == 0 else slice(None)   # do not duplicate the 0 line
            new_coor = numpy.concatenate((coor, -coor[keep][::-1]))
            new_Z = numpy.concatenate((Z, numpy.flip(numpy.take(Z, numpy.arange(coor.size)[keep], axis=axis), axis=axis)), axis=axis)
        else:                 # data on the positive side: mirror to the negative side
            keep = slice(1, None) if coor[0] == 0 else slice(None)
            new_coor = numpy.concatenate((-coor[keep][::-1], coor))
            new_Z = numpy.concatenate((numpy.flip(numpy.take(Z, numpy.arange(coor.size)[keep], axis=axis), axis=axis), Z), axis=axis)

        if axis == 0: self.x_interpolated = new_coor
        else:         self.y_interpolated = new_coor
        self.Z_INTERPOLATED = new_Z

    def triangulate(self, force=False):
        """
        Delaunay triangulation of the deformed (x, y) nodes, used by :meth:`interpolate` for point clouds.

        Parameters
        ----------
        force : bool, optional
            OASYS hdf5 files come on a regular grid and are not triangulated (it is not needed and is slow for
            large grids) unless force=True.
        """
        if self.file_in_type == 2 and not force:
            print("Regular grid (OASYS surface file): triangulation not needed, skipped.")
            return
        self.triPi = numpy.array([self.Xdeformed(), self.Ydeformed()]).transpose()
        self.tri = spatial.Delaunay(self.triPi)

    def plot_triangulation(self,show=True):
        """Plot the triangulation and the nodes (matplotlib); returns the figure."""
        import matplotlib.pyplot as plt
        if self.tri is None: self.triangulate(force=True)
        fig = plt.figure()
        plt.triplot(self.Xdeformed(), self.Ydeformed() , self.tri.simplices.copy())
        plt.plot(self.Xdeformed(), self.Ydeformed(), "or", label = "Data")
        plt.grid()
        plt.legend()
        plt.title("triangulation")
        plt.xlabel("x")
        plt.ylabel("y")
        if show:
            plt.show()
        return fig

    def get_Xinterpolated_mesh(self):
        """2D array of the x coordinates of the regular grid, X[ix, iy]."""
        return numpy.outer(self.x_interpolated,numpy.ones_like(self.y_interpolated))

    def get_Yinterpolated_mesh(self):
        """2D array of the y coordinates of the regular grid, Y[ix, iy]."""
        return numpy.outer(numpy.ones_like(self.x_interpolated),self.y_interpolated)

    def interpolate_grid(self, nx, ny, remove_nan=0):
        """
        Interpolate to a centred grid of exactly nx x ny points (recommended entry point).

        Point clouds: interpolates on nx+2 x ny+2 points over the data limits and removes one point on each
        side (the rim is where the interpolation is less reliable or NaN).
        OASYS hdf5 files: interpolates on nx x ny points (no interpolation if nx or ny <= 0); the borders
        are removed only if the file itself contains NaNs.

        Parameters
        ----------
        nx, ny : int
            Number of points along axis 0 (X) and axis 1 (Y).
        remove_nan : int, optional
            Values outside the convex hull of the nodes (point clouds): 0=NaN, 1=minimum height, 2=zero.
        """
        if self.file_in_type == 2:
            self.interpolate(nx, ny, remove_nan=remove_nan)
            if self.does_interpolated_have_nan():
                self.remove_borders_in_interpolated_data()
        else:
            self.interpolate(nx + 2, ny + 2, remove_nan=remove_nan)
            self.remove_borders_in_interpolated_data()

    def interpolate(self,nx,ny,remove_nan=0):
        """
        Interpolate on nx x ny points spanning the limits of the deformed nodes (lower level than
        :meth:`interpolate_grid`, which also handles the borders).

        Point clouds: cubic interpolation of the triangulated nodes (scipy griddata).
        OASYS hdf5 files: bicubic spline of the regular grid; nx or ny <= 0 means no interpolation.

        Parameters
        ----------
        nx, ny : int
            Number of points along axis 0 (X) and axis 1 (Y).
        remove_nan : int, optional
            Values outside the convex hull of the nodes (point clouds): 0=NaN, 1=minimum height, 2=zero.
        """
        if self.file_in_type == 2:
            if nx <= 0 or ny <=0:
                # nothing to do
                print("Skip interpolation")
            else:
                lim = self.get_limits_deformed()
                print("interpolation limits: xmin, xmax, ymin, ymax", lim[0:4])
                x_interpolated = numpy.linspace(lim[0], lim[1], nx)
                y_interpolated = numpy.linspace(lim[2], lim[3], ny)

                f = RectBivariateSpline(self.x_interpolated,
                             self.y_interpolated,
                             self.Z_INTERPOLATED)
                Z_INTERPOLATED = f(x_interpolated, y_interpolated)

                print("original dimensions for interpolation", self.Z_INTERPOLATED.shape)
                print("interpolated dimensions", Z_INTERPOLATED.shape)

                # overwrite interpolated data
                self.x_interpolated = x_interpolated
                self.y_interpolated = y_interpolated
                self.Z_INTERPOLATED = Z_INTERPOLATED
        else:
            if self.tri is None:
                self.triangulate()

            lim = self.get_limits_deformed()
            print("interpolation limits: xmin, xmax, ymin, ymax", lim[0:4])
            self.x_interpolated = numpy.linspace(lim[0],lim[1],nx)
            self.y_interpolated = numpy.linspace(lim[2],lim[3],ny)

            X_INTERPOLATED =  self.get_Xinterpolated_mesh()
            Y_INTERPOLATED =  self.get_Yinterpolated_mesh()

            self.P = numpy.array([X_INTERPOLATED.flatten(), Y_INTERPOLATED.flatten() ]).transpose()

            if remove_nan ==2:
                self.Z_INTERPOLATED = interpolate.griddata(self.triPi, self.Zdeformed(), self.P, rescale=True, method = "cubic", fill_value=0.0 ).reshape([nx,ny])
            elif remove_nan ==1:
                self.Z_INTERPOLATED = interpolate.griddata(self.triPi, self.Zdeformed(), self.P, rescale=True, method = "cubic", fill_value=self.Zdeformed().min() ).reshape([nx,ny])
            elif remove_nan == 0:
                self.Z_INTERPOLATED = interpolate.griddata(self.triPi, self.Zdeformed(), self.P, rescale=True, method="cubic").reshape([nx, ny])

    def plot_interpolated(self, show=True):
        """Contour plot of the interpolated surface with the nodes (matplotlib); returns the figure."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.contourf(self.get_Xinterpolated_mesh(), self.get_Yinterpolated_mesh(), self.Z_INTERPOLATED, 50, cmap = mpl.cm.jet)
        plt.colorbar()
        plt.contour(self.get_Xinterpolated_mesh(), self.get_Yinterpolated_mesh(), self.Z_INTERPOLATED, 20, colors = "k")
        plt.plot(self.Xdeformed(), self.Ydeformed(), "or", label = "Data")
        plt.legend()
        # plt.title = "Interpolated"  <---- THIS MAKES ERROR IN THE NEXT PLOT!!!!!!!!!!!!!!!!!
        plt.grid()
        if show:
            plt.show()
        return fig

    def plot_surface_image(self,invert_axes_names=False):
        """Image plot of the interpolated surface (srxraylib); invert_axes_names only changes the axis labels."""
        from srxraylib.plot.gol import plot_image
        if invert_axes_names:
            plot_image(self.Z_INTERPOLATED,self.x_interpolated,self.y_interpolated ,title="file: %s, axes names INVERTED from ANSYS"%self.filename,
                       xtitle="Y (%d pixels, max:%f)"%(self.x_interpolated.size,self.x_interpolated.max()),
                       ytitle="X (%d pixels, max:%f)"%(self.y_interpolated.size,self.y_interpolated.max()) )
        else:
            plot_image(self.Z_INTERPOLATED,self.x_interpolated,self.y_interpolated,title="file: %s, axes as in ANSYS"%self.filename,
                       xtitle="X (%d pixels, max:%f)"%(self.x_interpolated.size,self.x_interpolated.max()),
                       ytitle="Y (%d pixels, max:%f)"%(self.y_interpolated.size,self.y_interpolated.max()) )

    def does_interpolated_have_nan(self):
        """True if the interpolated surface contains NaN values."""
        return numpy.isnan(self.Z_INTERPOLATED).sum() > 0

    def remove_borders_in_interpolated_data(self):
        """Remove one point on each side of the interpolated grid (keeps the grid centred)."""
        self.x_interpolated = self.x_interpolated[1:-1].copy()
        self.y_interpolated = self.y_interpolated[1:-1].copy()
        self.Z_INTERPOLATED = self.Z_INTERPOLATED[1:-1,1:-1].copy()

    def detrend_straight_line(self,axis=0,fitting_domain_ratio=0.5):
        """
        Remove a straight line (tilt) fitted to the central profile along the given axis.

        Parameters
        ----------
        axis : int, optional
            0: fit along X (profile at the central Y), 1: fit along Y (profile at the central X).
        fitting_domain_ratio : float, optional
            The fit uses the points with abs(coordinate) <= fitting_domain_ratio [m] (same meaning as in
            :meth:`detrend_best_circle`).
        """
        if axis == 0:
            xm = self.x_interpolated.copy()
            zm = self.Z_INTERPOLATED[:,self.y_interpolated.size//2]
        elif axis == 1:
            xm = self.y_interpolated.copy()
            zm = self.Z_INTERPOLATED[self.x_interpolated.size // 2, :]

        zm.shape = -1

        icut = numpy.argwhere(numpy.abs(xm) <= fitting_domain_ratio)
        if len(icut) <=5:
            raise Exception("Not enough points for fitting.")

        xcut = xm[icut]
        zmcut = zm[icut]

        xcut.shape = -1
        zmcut.shape = -1

        print("Fitting interval: [%g,%g]" % (xcut[0],xcut[-1]))

        coeff = numpy.polyfit(xcut.copy(), zmcut.copy(), deg=1)

        zfit = coeff[0] * xm  + coeff[1]

        print("Detrending straight line (axis=%d): zfit = %g * coordinate + %g " % (axis, coeff[0], coeff[1]))

        if axis ==0:
            for i in range(self.Z_INTERPOLATED.shape[1]):
                self.Z_INTERPOLATED[:,i] -= zfit
        elif axis == 1:
            for i in range(self.Z_INTERPOLATED.shape[0]):
                self.Z_INTERPOLATED[i,:] -= zfit

    def detrend_best_circle(self,axis=0,fitting_domain_ratio=0.5):
        """
        Remove the best circle (curvature) of the central profile along the given axis.

        The radius is obtained from a straight-line fit of the slopes of the central profile; the circle
        sag, centred at coordinate 0, is subtracted from every profile along that axis.

        Parameters
        ----------
        axis : int, optional
            0: along X (profile at the central Y), 1: along Y (profile at the central X).
        fitting_domain_ratio : float, optional
            The fit uses the points with abs(coordinate) <= fitting_domain_ratio [m].

        Returns
        -------
        tuple
            (coordinates, subtracted circle sag) along the axis.
        """
        if axis == 0:
            xm = self.x_interpolated.copy()
            zm = self.Z_INTERPOLATED[:,self.y_interpolated.size//2]
        elif axis == 1:
            xm = self.y_interpolated.copy()
            zm = self.Z_INTERPOLATED[self.x_interpolated.size // 2, :]

        zm.shape = -1

        icut = numpy.argwhere(numpy.abs(xm) <= fitting_domain_ratio)
        if len(icut) <=5:
            raise Exception("Not enough points for fitting.")

        xcut = xm[icut]
        zmcut = zm[icut]

        xcut.shape = -1
        zmcut.shape = -1

        print("Fitting interval: [%g,%g] (using %d points)" % (xcut[0],xcut[-1],xcut.size))

        coeff = numpy.polyfit(xcut, numpy.gradient(zmcut,xcut), deg=1)

        radius = 1 / coeff[0]
        print("Detrending straight line on slopes (axis=%d): slope fit = %g * coordinate + %g " % (axis, coeff[0], coeff[1]))
        print("Radius of curvature: %g m" % radius)

        # circle sag: R - sqrt(R^2 - x^2) (R >= 0) or R + sqrt(R^2 - x^2) (R < 0), written in the
        # algebraically equivalent form x^2 / (R + sign(R) sqrt(R^2 - x^2)), which does not lose
        # precision for large radii
        sign = 1.0 if radius >= 0 else -1.0
        zfit = xm ** 2 / (radius + sign * numpy.sqrt(radius ** 2 - xm ** 2))

        if axis ==0:
            for i in range(self.Z_INTERPOLATED.shape[1]):
                self.Z_INTERPOLATED[:,i] -= zfit
        elif axis == 1:
            for i in range(self.Z_INTERPOLATED.shape[0]):
                self.Z_INTERPOLATED[i,:] -= zfit

        return xm, zfit

    def reset_height_to_minimum(self):
        """Shift the heights so that the minimum is 0."""
        self.Z_INTERPOLATED -= self.Z_INTERPOLATED.min()

    def reset_height_to_central_value(self):
        """Shift the heights so that the height at the central grid point is 0."""
        self.Z_INTERPOLATED -= self.Z_INTERPOLATED[self.Z_INTERPOLATED.shape[0]//2,self.Z_INTERPOLATED.shape[1]//2]

    def write_h5_surface(self,filename='presurface.hdf5',invert_axes_names=False):
        """
        Write the interpolated surface as an OASYS surface file (hdf5).

        The file stores Z[iy, ix] (the transpose of Z_INTERPOLATED) with datasets X, Y, Z.

        Parameters
        ----------
        filename : str, optional
            Output file name (overwritten if it exists).
        invert_axes_names : bool, optional
            If True, X and Y are swapped: the file X is the ANSYS Y and the file Y is the ANSYS X.
        """
        if invert_axes_names:
            write_surface_file(self.Z_INTERPOLATED, self.y_interpolated, self.x_interpolated, filename, overwrite=True)
        else:
            write_surface_file(self.Z_INTERPOLATED.T, self.x_interpolated, self.y_interpolated, filename, overwrite=True)
        print("write_h5_surface: File for OASYS " + filename + " written to disk.")

    def gaussian_filter(self,sigma_axis0=10,sigma_axis1=10):
        """
        Smooth the interpolated surface with a Gaussian filter.

        Parameters
        ----------
        sigma_axis0, sigma_axis1 : float, optional
            Standard deviation of the Gaussian kernel along axis 0 (X) and axis 1 (Y), in pixels.
        """
        from scipy.ndimage import gaussian_filter
        self.Z_INTERPOLATED = gaussian_filter(self.Z_INTERPOLATED, (sigma_axis0,sigma_axis1),
                        order=0, output=None, mode='nearest', cval=0.0, truncate=4.0)

def surface_plot(xs,ys,zs):
    """3D scatter plot of points (e.g. the output of FEA_File.get_deformed())."""
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":
    from srxraylib.plot.gol import plot_image

    # demo: a synthetic OASYS surface file (Z[iy, ix] = tilt along x + circle along y)
    x = numpy.linspace(-0.05, 0.05, 101)
    y = numpy.linspace(-0.01, 0.01, 41)
    write_surface_file(1e-6 * (x[None, :] / 0.05) + y[:, None] ** 2 / (2 * 50.0), x, y, "FEA_File_demo.h5", overwrite=True)

    fea_file_object = FEA_File("FEA_File_demo.h5", file_in_type=2, load=True)

    fea_file_object.interpolate(0, 0, remove_nan=0)
    print(fea_file_object.x_interpolated.shape,
          fea_file_object.y_interpolated.shape,
          fea_file_object.Z_INTERPOLATED.shape,)
    plot_image(fea_file_object.Z_INTERPOLATED, fea_file_object.x_interpolated, fea_file_object.y_interpolated,
               aspect='auto', title="No interpolation", show=0)

    fea_file_object.interpolate(70, 800, remove_nan=0)
    print(fea_file_object.x_interpolated.shape,
          fea_file_object.y_interpolated.shape,
          fea_file_object.Z_INTERPOLATED.shape,)

    plot_image(fea_file_object.Z_INTERPOLATED, fea_file_object.x_interpolated, fea_file_object.y_interpolated,
               aspect='auto', title="Interpolated")

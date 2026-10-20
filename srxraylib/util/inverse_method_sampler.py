"""
 Classes for creating random points following a given numeric distribution using the inverse method. Covers 1D, 2D and 3D sampling.

 See tests for examples of use.
"""

__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"
__date__ = "27/08/2018"

import numpy

def _cdf_trapezoid(pdf, axis=-1):
    """
    Cumulative distribution function at the abscissas of the pdf, computed with the trapezoidal rule along an axis,
    and normalized to one (unless all the values are zero).

    Note: the cdf computed with numpy.cumsum(pdf) corresponds to the right edges of the bins, and interpolating it at
    the abscissas shifts the sampled points by half a bin. The trapezoidal cdf is centered at the abscissas.

    Parameters
    ----------
    pdf : numpy array
        The (non-normalized) pdf at equally spaced abscissas.
    axis : int, optional
        The axis along which the cdf is computed.

    Returns
    -------
    numpy array
        The cdf, with the same shape as pdf, starting at zero.
    """
    pdf = numpy.moveaxis(numpy.asarray(pdf, dtype=float), axis, -1)
    cdf = numpy.zeros_like(pdf)
    cdf[..., 1:] = numpy.cumsum(0.5 * (pdf[..., 1:] + pdf[..., :-1]), axis=-1)
    norm = cdf[..., -1:].copy()
    norm[norm == 0] = 1.0
    cdf /= norm
    return numpy.moveaxis(cdf, -1, axis)


def _nearest_index(ival, idelta, size, cdf_last=None):
    # Index of the grid point closest to the sampled point x[ival] + idelta * dx (0 <= idelta < 1), used to select
    # the conditional cdf. cdf_last (optional) contains the last value of the conditional cdfs (zero for an empty
    # row, e.g., a pdf vanishing at a border): an empty row is replaced by its neighbor in the interval.
    i_near = int(min(ival + (1 if idelta >= 0.5 else 0), size - 1))
    if cdf_last is not None and cdf_last[i_near] == 0:
        i_other = int(min(ival + 1, size - 1)) if i_near == ival else int(ival)
        if cdf_last[i_other] != 0: i_near = i_other
    return i_near


class Sampler1D(object):

    """
    Constructor.

    Parameters
    ----------
    pdf : numpy array
        1D input probability distrubution function.
    pdf_x : numpy array
        the abscissas of the odf.
    cdf_interpolation_factor : float, optional
        interpolation factor for calculating the cdf (1 makes no interpolation)/

    """
    def __init__(self, pdf, pdf_x=None, cdf_interpolation_factor=1):
        self._pdf = pdf
        if pdf_x is None:
            self._set_default_pdf_x()
        else:
            self._pdf_x = pdf_x


        self._cdf_interpolation_factor = cdf_interpolation_factor
        self._cdf_calculate()  # defines self._cdf and self._cdf_x

    def pdf(self):
        """
        Gets the array with probability distribution function (pdf).

        Returns
        -------
        numpy array
            The pdf array (referenced, not copied).
        """
        return self._pdf

    def abscissas(self):
        """
        Gets the abscissas array.

        Returns
        -------
        numpy array
            The abscissas array (referenced, not copied).
        """
        return self._pdf_x

    def cdf(self):
        """
        Gets the cumulative distribution function (cdf).

        Returns
        -------
        numpy array
            The cdf (referenced, not copied).

        """
        return self._cdf

    def cdf_abscissas(self):
        """
        Gets the abscissas of the cumulative distribution function (cdf).

        Returns
        -------
        numpy array
            The cdf abscissas (referenced, not copied).

        """
        return self._cdf_x

    def get_sampled(self, random_in_0_1):
        """
        Return an array with sampled points.

        Parameters
        ----------
        random_in_0_1  : float or numpy array
            Points sampled in a uniform interval.

        Returns
        -------
        numpy array
            the points sampled with the current pdf. The number of points is equal to the dimension of random_in_0_1.

        """
        y = numpy.array(random_in_0_1)

        if y.size > 1:
            x_rand_array = numpy.zeros_like(random_in_0_1)
            for i,cdf_rand in enumerate(random_in_0_1):
                ival,idelta,pendent = self._get_index(cdf_rand)
                x_rand_array[i] = self._pdf_x[ival] + idelta*(self._pdf_x[1]-self._pdf_x[0])
            return x_rand_array
        else:
            ival,idelta,pendent = self._get_index(random_in_0_1)
            return self._pdf_x[ival] + idelta*(self._pdf_x[1]-self._pdf_x[0])

    def get_sampled_and_histogram(self, random_in_0_1, bins=51, range=None):
        """
        Return an array with sampled points and the histogram.

        Parameters
        ----------
        random_in_0_1  : float or numpy array
            Points sampled in a uniform interval.
            bins : int, optional
                Number of bins
            range : list or tuple
                [min, max] the histogram limits.

        Returns
        -------
        tuple
            (s1, h, bin_edges)
            s1: the points sampled with the current pdf. The number of points is equal to the dimension of random_in_0_1,
            h: the array with the histogram values at the bin edges,
            bin_edges: the bin edges.

        """
        s1 = self.get_sampled(random_in_0_1)
        if range is None:
            range = [self._pdf_x.min(),self._pdf_x.max()]
        #
        # histogram
        #
        h, bin_edges = numpy.histogram(s1,bins=bins,range=range)
        return s1, numpy.array(h), numpy.array(bin_edges)

    def get_n_sampled_points(self, npoints, seed=None):
        """
        Returns a given number points sampled points sampled with the pdf.

        Parameters
        ----------
        npoints : int
            The number of points.
        seed : int, optional
            The seed (numpy generator is initialized with numpy.random.default_rng(seed))

        Returns
        -------
        numpy array
            The sampled points.

        """
        if not seed is None:
            rng = numpy.random.default_rng(seed)
            cdf_rand_array = rng.random(npoints)
        else:
            cdf_rand_array = numpy.random.random(npoints)

        return self.get_sampled(cdf_rand_array)

    def get_n_sampled_points_and_histogram(self, npoints, bins=51, range=None, seed=None):
        """
        Returns a given number points sampled points sampled with the pdf and the histogram.

        Parameters
        ----------
        npoints : int
            The number of points.
        seed : int, optional
            The seed (numpy generator is initialized with numpy.random.default_rng(seed))
        bins : int, optional
            Number of bins
        range : list or tuple
            [min, max] the histogram limits.

        Returns
        -------
        tuple
            (s1, h, bin_edges)
            s1: the points sampled with the current pdf. The number of points is equal to the dimension of random_in_0_1,
            h: the array with the histogram values at the bin edges,
            bin_edges: the bin edges.

        """
        if not seed is None:
            rng = numpy.random.default_rng(seed)
            cdf_rand_array = rng.random(npoints)
        else:
            cdf_rand_array = numpy.random.random(npoints)
        return self.get_sampled_and_histogram(cdf_rand_array,bins=bins,range=range)

    def _set_default_pdf_x(self):
        self._pdf_x = numpy.arange(self._pdf.size)

    def _cdf_calculate(self):
        if self._cdf_interpolation_factor != 1:
            xx = numpy.linspace(self.abscissas()[0],self.abscissas()[-1],self.abscissas().size*2)
            yy = numpy.interp(xx,self.abscissas(),self.pdf())
            self._cdf = _cdf_trapezoid(yy)
            self._cdf_x = xx
        else:
            self._cdf = _cdf_trapezoid(self._pdf)
            self._cdf_x = self._pdf_x.copy()

    def _get_index(self,edge):
        try:
            ix = numpy.nonzero(self._cdf >= edge)[0][0]
        except:
            ix = 0

        if ix > 0:
            ix -= 1

        if ix >= (self._cdf.size - 1):
            pendent = 0.0
            delta = 0.0
        else:
            pendent = self._cdf[ix + 1] - self._cdf[ix]
            delta = (edge - self._cdf[ix]) / pendent if pendent != 0 else 0.0

        return ix//self._cdf_interpolation_factor,delta,pendent


class Sampler2D(object):
    """
    Constructor.

    Parameters
    ----------
    pdf : numpy array
        the 2D pdf.
    pdf_x0 : numpy array
        A 1D array with the abscissas for axis 0.
    pdf_x1 : numpy array
        A 1D array with the abscissas for axis 1.
    """

    def __init__(self, pdf, pdf_x0=None, pdf_x1=None):
        self._pdf = pdf
        if pdf_x0 is None:
            self._pdf_x0 = numpy.arange(self._pdf.shape[0])
        else:
            self._pdf_x0 = pdf_x0

        if pdf_x1 is None:
            self._pdf_x1 = numpy.arange(self._pdf.shape[1])
        else:
            self._pdf_x1 = pdf_x1

        self._cdf2,self._cdf1 = self._cdf_calculate()

    def pdf(self):
        """
        Gets the array with probability distribution function (pdf).

        Returns
        -------
        numpy array
            The pdf array (referenced, not copied).
        """
        return self._pdf

    def cdf(self):
        """
        Gets the array with cumulated distribution function (cdf).

        Returns
        -------
        numpy array
            The cdf array (referenced, not copied).
        """
        return self._cdf2,self._cdf1

    def abscissas(self):
        """
        Gets the abscisas arrays.

        Returns
        -------
        tuple
            (x0, x1) The arrays for axes 0 and 1.
        """
        return self._pdf_x0,self._pdf_x1

    def get_sampled(self, random0, random1):
        """
        Samples a point or multiple points in 2D (two coordinates) following the given pdf.

        Parameters
        ----------
        random0 : float or numpy array
            The 1D array with values unifiormly samples in [0,1]
        random1 : float or numpy array
            The 1D array with values unifiormly samples in [0,1]

        Returns
        -------
        tuple
            (x,y) the coordinates x (float or array) and y (float or array) of the sampled point(s).

        """
        y0 = numpy.array(random0)
        y1 = numpy.array(random1)

        if y0.size > 1:
            x0_rand_array = numpy.zeros_like(y0)
            x1_rand_array = numpy.zeros_like(y1)

            for i,cdf_rand0 in enumerate(y0):
                ival,idelta,pendent = self._get_index0(cdf_rand0)
                x0_rand_array[i] = self._pdf_x0[ival] + idelta*(self._pdf_x0[1]-self._pdf_x0[0])
                ival1,idelta1,pendent1 = self._get_index1(y1[i],_nearest_index(ival,idelta,self._pdf_x0.size,
                                                                               self._cdf2[:, -1]))
                x1_rand_array[i] = self._pdf_x1[ival1] + idelta1*(self._pdf_x1[1]-self._pdf_x1[0])
            return x0_rand_array, x1_rand_array
        else:
            pass # TODO make scalar case

    def get_sampled_x2(self, random0, random10, random11):
        """
        Samples a point or multiple points in 2D (two coordinates) following the given pdf.
        It samples one point in axis 0 and two points on the axis 1.

        Parameters
        ----------
        random0 : float or numpy array
            The 1D array with values unifiormly samples in [0,1]
        random10 : float or numpy array
            The 1D array with values unifiormly samples in [0,1]
        random11 : float or numpy array
            The 1D array with values unifiormly samples in [0,1]

        Returns
        -------
        tuple
            (x, y0, y1) the coordinates x (float or array) on the axis 0 and and y0, y1 (float or array) of the sampled
            point(s) in axis 1.

        """
        y0 = numpy.array(random0)
        y10 = numpy.array(random10)
        y11 = numpy.array(random11)

        if y0.size > 1:
            x0_rand_array = numpy.zeros_like(y0)
            x10_rand_array = numpy.zeros_like(y10)
            x11_rand_array = numpy.zeros_like(y11)

            for i,cdf_rand0 in enumerate(y0):
                ival, idelta, pendent = self._get_index0(cdf_rand0)
                x0_rand_array[i] = self._pdf_x0[ival] + idelta*(self._pdf_x0[1]-self._pdf_x0[0])
                i0 = _nearest_index(ival, idelta, self._pdf_x0.size, self._cdf2[:, -1])
                ival10, idelta10, pendent10 = self._get_index1(y10[i], i0)
                x10_rand_array[i] = self._pdf_x1[ival10] + idelta10 * (self._pdf_x1[1] - self._pdf_x1[0])
                ival11, idelta11, pendent11 = self._get_index1(y11[i], i0)
                x11_rand_array[i] = self._pdf_x1[ival11] + idelta11 * (self._pdf_x1[1] - self._pdf_x1[0])
            return x0_rand_array, x10_rand_array, x11_rand_array
        else:
            pass # TODO make scalar case

    def get_n_sampled_points(self, npoints, seed=None):
        """
        Samples n points (two coordinates) following the given pdf.

        Parameters
        ----------
        npoints : int
            The number of points.
        seed : int, optional
            The seed (numpy generator is initialized with numpy.random.default_rng(seed))


        Returns
        -------
        tuple
            (x,y) the coordinates x (float or array) and y (float or array) of the sampled point(s).

        """
        if not seed is None:
            rng = numpy.random.default_rng(seed)
            cdf_rand_array0 = rng.random(npoints)
            cdf_rand_array1 = rng.random(npoints)
        else:
            cdf_rand_array0 = numpy.random.random(npoints)
            cdf_rand_array1 = numpy.random.random(npoints)

        return self.get_sampled(cdf_rand_array0, cdf_rand_array1)

    def get_n_sampled_points_x2(self, npoints, seed=None):
        """
        Samples n points (two coordinates, two times the second axis) following the given pdf.

        Parameters
        ----------
        npoints : int
            The number of points.
        seed : int, optional
            The seed (numpy generator is initialized with numpy.random.default_rng(seed))


        Returns
        -------
        tuple
            (x, y0, y1) the coordinates x (float or array) on the axis 0 and and y0, y1 (float or array) of the sampled
            point(s) on axis 1.

        """
        if not seed is None:
            rng = numpy.random.default_rng(seed)
            cdf_rand_array0 = rng.random(npoints)
            cdf_rand_array1 = rng.random(npoints)
            cdf_rand_array2 = rng.random(npoints)
        else:
            cdf_rand_array0 = numpy.random.random(npoints)
            cdf_rand_array1 = numpy.random.random(npoints)
            cdf_rand_array2 = numpy.random.random(npoints)

        return self.get_sampled_x2(cdf_rand_array0, cdf_rand_array1, cdf_rand_array2)

    def _cdf_calculate(self):
        pdf2 = self._pdf
        pdf1 = pdf2.sum(axis=1)

        cdf2 = _cdf_trapezoid(pdf2, axis=1)   # conditional cdf along axis 1 for each point of axis 0
        cdf1 = _cdf_trapezoid(pdf1)           # marginal cdf along axis 0

        return cdf2,cdf1

    def _get_index0(self,edge):

        try:
            ix = numpy.nonzero(self._cdf1 >= edge)[0][0]
        except:
            ix = 0

        if ix > 0:
            ix -= 1

        if ix == (self._cdf1.size-1):
            pendent = 0.0
            delta = 0.0
        else:
            pendent = self._cdf1[ix+1] - self._cdf1[ix]
            delta = (edge - self._cdf1[ix]) / pendent if pendent != 0 else 0.0
        return ix,delta,pendent

    def _get_index1(self,edge,index0):

        try:
            ix = numpy.nonzero(self._cdf2[index0, :] >= edge)[0][0]
        except:
            ix = 0

        if ix > 0:
            ix -= 1

        if ix == (self._cdf2[index0,:].size-1):
            pendent = 0.0
            delta = 0.0
        else:
            pendent = self._cdf2[index0,(ix+1)] - self._cdf2[index0,ix]
            delta = (edge - self._cdf2[index0,ix]) / pendent if pendent != 0 else 0.0
        return ix,delta,pendent


class Sampler3D(object):
    """
    Constructor.

    Parameters
    ----------
    pdf : numpy array
        The 3D pdf.
    pdf_x0 : numpy array
        The abscissas for axis 0.
    pdf_x1 : numpy array
        The abscissas for axis 1.
    pdf_x2 : numpy array
        The abscissas for axis 2.

    """
    def __init__(self, pdf, pdf_x0=None, pdf_x1=None, pdf_x2=None):
        self._pdf = pdf
        if pdf_x0 is None:
            self._pdf_x0 = numpy.arange(self._pdf.shape[0])
        else:
            self._pdf_x0 = pdf_x0

        if pdf_x1 is None:
            self._pdf_x1 = numpy.arange(self._pdf.shape[1])
        else:
            self._pdf_x1 = pdf_x1

        if pdf_x2 is None:
            self._pdf_x2 = numpy.arange(self._pdf.shape[2])
        else:
            self._pdf_x2 = pdf_x2

        self._cdf3,self._cdf2,self._cdf1 = self._cdf_calculate()

    def pdf(self):
        """
        Gets the array with probability distribution function (pdf).

        Returns
        -------
        numpy array
            The pdf array (referenced, not copied).
        """
        return self._pdf

    def cdf(self):
        """
        Gets the array with cumulative distribution function (cdf).

        Returns
        -------
        numpy array
            The cdf array (referenced, not copied).
        """
        return self._cdf3,self._cdf2,self._cdf1

    def abscissas(self):
        """
        Gets the arrays with the abscissas.

        Returns
        -------
        tuple
            (x0, x1, x2) The arrays with the abscissas for axes 1, 2, and 3.

        """
        return self._pdf_x0,self._pdf_x1,self._pdf_x2

    def get_sampled(self, random0, random1, random2):
        """
        Get sampled 3D points.

        Parameters
        ----------
        random0 : float or numpy array
            The points or points sampled uniformly in a [0,1] interval.
        random1 : float or numpy array
            The points or points sampled uniformly in a [0,1] interval.
        random2 : float or numpy array
            The points or points sampled uniformly in a [0,1] interval.

        Returns
        -------
        tuple
            (x0,x1,x2) the coordinates (float or array) of the sampled points.

        """
        y0 = numpy.array(random0)
        y1 = numpy.array(random1)
        y2 = numpy.array(random2)

        if y0.size > 1:
            x0_rand_array = numpy.zeros_like(y0)
            x1_rand_array = numpy.zeros_like(y1)
            x2_rand_array = numpy.zeros_like(y2)
            for i,cdf_rand0 in enumerate(y0):
                ival,idelta,pendent = self._get_index0(cdf_rand0)
                x0_rand_array[i] = self._pdf_x0[ival] + idelta*(self._pdf_x0[1]-self._pdf_x0[0])

                i0 = _nearest_index(ival, idelta, self._pdf_x0.size, self._cdf2[:, -1])
                ival1,idelta1,pendent1 = self._get_index1(y1[i],i0)
                x1_rand_array[i] = self._pdf_x1[ival1] + idelta1*(self._pdf_x1[1]-self._pdf_x1[0])

                i1 = _nearest_index(ival1, idelta1, self._pdf_x1.size, self._cdf3[i0, :, -1])
                ival2,idelta2,pendent2 = self._get_index2(y2[i],i0,i1)
                x2_rand_array[i] = self._pdf_x2[ival2] + idelta2*(self._pdf_x2[1]-self._pdf_x2[0])

            return x0_rand_array,x1_rand_array,x2_rand_array
        else:
            pass #TODO do the scalar case


    def get_n_sampled_points(self, npoints, seed=None):
        """
        Get a given number of sampled 3D points.

        Parameters
        ----------
        npoints : int
            The number of points.
        seed : int, optional
            The seed (numpy generator is initialized with numpy.random.default_rng(seed))


        Returns
        -------
        tuple
            (x,y,z) the coordinates x (float or array), y (float or array) and z (float or array) of the sampled point(s).

        """
        if not seed is None:
            rng = numpy.random.default_rng(seed)
            cdf_rand_array0 = rng.random(npoints)
            cdf_rand_array1 = rng.random(npoints)
            cdf_rand_array2 = numpy.random.random(npoints)
        else:
            cdf_rand_array0 = numpy.random.random(npoints)
            cdf_rand_array1 = numpy.random.random(npoints)
            cdf_rand_array2 = numpy.random.random(npoints)

        return self.get_sampled(cdf_rand_array0, cdf_rand_array1, cdf_rand_array2)


    def _cdf_calculate(self):

        pdf3 = self._pdf
        pdf2 = pdf3.sum(axis=2)
        pdf1 = pdf2.sum(axis=1)

        cdf3 = _cdf_trapezoid(pdf3, axis=2)   # conditional cdf along axis 2 for each point of axes 0, 1
        cdf2 = _cdf_trapezoid(pdf2, axis=1)   # conditional cdf along axis 1 for each point of axis 0
        cdf1 = _cdf_trapezoid(pdf1)           # marginal cdf along axis 0

        return cdf3,cdf2,cdf1

    def _get_index0(self,edge):

        try:
            ix = numpy.nonzero(self._cdf1 > edge)[0][0]
        except:
            ix = 0
        if ix > 0:
            ix -= 1
        if ix == (self._cdf1.size-1):
            pendent = 0.0
            delta = 0.0
        else:
            pendent = self._cdf1[ix+1] - self._cdf1[ix]
            delta = (edge - self._cdf1[ix]) / pendent if pendent != 0 else 0.0
        return ix,delta,pendent

    def _get_index1(self,edge,index0):
        try:
            ix = numpy.nonzero(self._cdf2[index0,:] > edge)[0][0]
        except:
            ix = 0
        if ix > 0:
            ix -= 1
        if ix == (self._cdf2[index0,:].size-1):
            pendent = 0.0
            delta = 0.0
        else:
            pendent = self._cdf2[index0,(ix+1)] - self._cdf2[index0,ix]
            delta = (edge - self._cdf2[index0,ix]) / pendent if pendent != 0 else 0.0
        return ix,delta,pendent


    def _get_index2(self,edge,index0,index1):
        try:
            ix = numpy.nonzero(self._cdf3[index0, index1, :] > edge)[0][0]
        except:
            ix = 0

        if ix > 0:
            ix -= 1
        if ix == (self._cdf3[index0,index1,:].size-1):
            pendent = 0.0
            delta = 0.0
        else:
            pendent = self._cdf3[index0,index1,(ix+1)] - self._cdf3[index0,index1,ix]
            delta = (edge - self._cdf3[index0,index1,ix]) / pendent if pendent != 0 else 0.0
        return ix,delta,pendent


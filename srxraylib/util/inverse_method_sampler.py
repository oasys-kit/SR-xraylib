"""
 Classes for creating random points following a given numeric distribution using the inverse method. Covers 1D, 2D and 3D sampling.

 See tests for examples of use.
"""

__authors__ = ["M Sanchez del Rio - ESRF ISDD Advanced Analysis and Modelling"]
__license__ = "MIT"
__date__ = "27/08/2018"

import numpy

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
        h, bin_edges = numpy.array(numpy.histogram(s1,bins=bins,range=range))
        return s1, h, bin_edges

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
            self._cdf = numpy.cumsum(yy)
            self._cdf_x = xx
        else:
            self._cdf = numpy.cumsum(self._pdf)
            self._cdf_x = self._pdf_x.copy()
        self._cdf -= self._cdf[0]
        if self._cdf.max() != 0.0:
            self._cdf /= self._cdf.max()

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
            delta = (edge - self._cdf[ix]) / pendent

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
                ival1,idelta1,pendent1 = self._get_index1(y1[i],ival+1)  # <==================== changed to ival+1
                x1_rand_array[i] = self._pdf_x1[ival1] + idelta1*(self._pdf_x1[1]-self._pdf_x1[0])
            return x0_rand_array, x1_rand_array
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


    def _cdf_calculate(self):
        pdf2 = self._pdf
        pdf1 = pdf2.sum(axis=1)

        cdf2 = numpy.zeros_like(pdf2)
        for i in range(cdf2.shape[0]):
            cdf2[i,:] = numpy.cumsum(pdf2[i,:])
            cdf2[i,:] -= cdf2[i,:][0]
            cdf2[i,:] = cdf2[i,:] / float(cdf2[i,:].max())

        cdf1 = numpy.cumsum(pdf1)
        cdf1 -= cdf1[0]
        cdf1 /= cdf1.max()

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
            delta = (edge - self._cdf1[ix]) / pendent
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
            delta = (edge - self._cdf2[index0,ix]) / pendent
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

                ival1,idelta1,pendent1 = self._get_index1(y1[i],ival+1) # <==================== TODO: test changed to ival+1
                x1_rand_array[i] = self._pdf_x1[ival1] + idelta1*(self._pdf_x1[1]-self._pdf_x1[0])

                ival2,idelta2,pendent2 = self._get_index2(y2[i],ival+1,ival1+1)  # <==================== TODO: test changed to ival+1,ival1+1
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

        cdf3 = numpy.zeros_like(pdf3)
        cdf2 = numpy.zeros_like(pdf2)

        for i in range(cdf3.shape[0]):
            for j in range(cdf3.shape[1]):
                tmp = numpy.cumsum(pdf3[i,j,:])
                cdf3[i, j, :] = tmp
                cdf3[i,j,:] -= cdf3[i,j,0]
                cdf3[i,j,:] = cdf3[i,j,:] / float(cdf3[i,j,:].max())
        #
        for i in range(cdf3.shape[0]):
            tmp = numpy.cumsum(pdf2[i,:])
            cdf2[i, :] = tmp
            cdf2[i,:] -= cdf2[i,0]
            cdf2[i,:] = cdf2[i,:] / float(cdf2[i,:].max())
        #
        #
        cdf1 = numpy.cumsum(pdf1)
        cdf1 -= cdf1[0]
        cdf1 /= cdf1.max()

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
            delta = (edge - self._cdf1[ix]) / pendent
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
            delta = (edge - self._cdf2[index0,ix]) / pendent
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
            delta = (edge - self._cdf3[index0,index1,ix]) / pendent
        return ix,delta,pendent


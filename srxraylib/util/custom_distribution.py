import numpy

class CustomDistribution(object):
    """
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = False, interpolation = False, transform = lambda x: x, seed=0):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform
        self.seed = seed

        #a pdf can not be negative
        assert(numpy.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = numpy.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = numpy.cumsum(self.pdf)
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]
    def __call__(self, N):
        if self.seed > 0: numpy.random.seed(self.seed)

        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = numpy.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = numpy.searchsorted(self.cdf, choice)
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = numpy.unravel_index(index, self.shape)
        index = numpy.vstack(index)
        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + numpy.random.uniform(size=index.shape)
        return self.transform(index)

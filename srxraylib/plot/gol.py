"""
GOL: Graphics in One Line

A collection of functions to create matplotlib plots in a single command line.

(This has been written after getting tired of polluting code with matplotlib instructions).
"""

__author__ = "Manuel Sanchez del Rio"
__contact__ = "srio@esrf.eu"
__copyright = "ESRF, 2016"

import numpy as np
try:
    import matplotlib.pylab as plt
except:
    raise ImportError("Please install matplotlib to allow graphics")


def set_qt():
    """
    Sets the 'correct' qt backend: plt.switch_backend("Qt5Agg").
    This is useful in python scripts making graphs inside oasys in some platforms.
    """
    try:
        plt.switch_backend("Qt5Agg")
    except:
        raise Exception("Failed to set matplotlib backend to Qt5Agg")

def plot_show():
    """
    executes matplotlib's plt.show().
    """
    plt.show()

def plot_image(*positional_parameters,title="TITLE",xtitle=r"X",ytitle=r"Y",
               xrange=None, yrange=None,
               cmap=None,aspect=None,show=1,
               add_colorbar=True,figsize=None):
    """
    Plots an image.

    Parameters
    ----------
    *positional_parameters : tuple(s)
        z [, x, y]
    title : str, optional
        The main title.
    xtitle : str, optional
        The X title.
    ytitle : str, optional
        The Y title.
    xrange : 2D tuple or list
        [min, max] values.
    yrange : 2D tuple or list
        [min, max] values.
    cmap : str, optional
        The matplotlin color map.
    aspect : str, optional
        The matplotlib aspect (e..g. 'auto')
    show : int, optional
        if 1 executes show_plot()
    add_colorbar : boolean, optional
        if True adds the color bar.
    figsize : tuple
        The matplotlib figure size.

    Returns
    -------
    tuple
        The matplotlib (fig, ax).

    """
    n_arguments = len(positional_parameters)
    if n_arguments == 1:
        z = positional_parameters[0]
        x = np.arange(0,z.shape[0])
        y = np.arange(0,z.shape[1])
    elif n_arguments == 2:
        z = positional_parameters[0]
        x = positional_parameters[1]
        y = positional_parameters[1]
    elif n_arguments == 3:
        z = positional_parameters[0]
        x = positional_parameters[1]
        y = positional_parameters[2]
    else:
        raise Exception("Bad number of inputs")


    fig = plt.figure(figsize=figsize)

    # cmap = plt.cm.Greys
    plt.imshow(z.T,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap=cmap,aspect=aspect)
    if add_colorbar:
        plt.colorbar()
    ax = fig.gca()
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    plt.title(title)

    plt.xlim( xrange )
    plt.ylim( yrange )

    if show:
        plt.show()

    return fig, ax


def plot_image_with_histograms(*positional_parameters,
                               title="", xtitle=r"X", ytitle=r"Y",
                               xrange=None, yrange=None,
                               cmap=None, aspect='auto', show=True,
                               add_colorbar=False, figsize=(8,8),
                               use_profiles_instead_histograms=False,
                               ):
    """
    Plots an image with an histogram.

    Parameters
    ----------
    *positional_parameters : tuple(s)
        z [, x, y]  Note that here, z array has Y in the first index z(y,x)!!!!
    title : str, optional
        The main title.
    xtitle : str, optional
        The X title.
    ytitle : str, optional
        The Y title.
    xrange : 2D tuple or list
        [min, max] values.
    yrange : 2D tuple or list
        [min, max] values.
    cmap : str, optional
        The matplotlin color map.
    aspect : str, optional
        The matplotlib aspect (e..g. 'auto')
    show : int, optional
        if 1 executes show_plot()
    add_colorbar : boolean, optional
        if True adds the color bar.
    figsize : tuple
        The matplotlib figure size.
    use_profiles_instead_histograms : boolean, optional
        If True, display the profiles at (0,0) instead of the histograms.

    Returns
    -------
    tuple
        The matplotlib objects (figure, axScatter, axHistx, axHisty).

    """

    if aspect is None: aspect == 'auto'

    n_arguments = len(positional_parameters)
    if n_arguments == 1:
        z = positional_parameters[0].T
        x = np.arange(0,z.shape[0])
        y = np.arange(0,z.shape[1])
    elif n_arguments == 2:
        z = positional_parameters[0].T
        x = positional_parameters[1]
        y = positional_parameters[1]
    elif n_arguments == 3:
        z = positional_parameters[0].T
        x = positional_parameters[1]
        y = positional_parameters[2]
    else:
        raise Exception("Bad number of inputs")

    if xrange is None:
        xrange = [x.min(),x.max()]

    if yrange is None:
        yrange = [y.min(),y.max()]


    figure = plt.figure(figsize=figsize)

    hfactor = 1.0
    vfactor = 1.0

    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    #
    #main plot
    #
    axScatter = figure.add_axes(rect_scatter)

    axScatter.set_xlabel(xtitle)
    axScatter.set_ylabel(ytitle)

    if aspect == 'equal':
        axScatter.set_aspect(aspect)

    axScatter.axis(xmin=hfactor*xrange[0],xmax=xrange[1])
    axScatter.axis(ymin=vfactor*yrange[0],ymax=yrange[1])



    axs = axScatter.pcolormesh(x,y,z,cmap=cmap)

    #
    #histograms
    #

    if aspect == 'equal':
        pos0 = axScatter.get_position()
        mm = np.min((pos0.height, pos0.width)) * 0.6
        axHistx = figure.add_axes([pos0.x0, pos0.y0 +pos0.height, pos0.width, mm], sharex=axScatter)
        axHisty = figure.add_axes([pos0.x0 + pos0.width, pos0.y0, mm * figsize[1] / figsize[0], pos0.height], sharey=axScatter)
    else:
        axHistx = figure.add_axes(rect_histx, sharex=axScatter)
        axHisty = figure.add_axes(rect_histy, sharey=axScatter)

    if use_profiles_instead_histograms:
        hx = z[z.shape[0]//2, :]
        hy = z[:, z.shape[1]//2]
    else:
        hx = z.sum(axis=0)
        hy = z.sum(axis=1)

    axHistx.plot(x,hx)
    axHisty.plot(hy,y)


    # tt = np.where(hx >= hx.max() * 0.5)
    # if hx[tt].size > 1:
    #     binSize = x[1] - x[0]
    #     print("FWHM X: ",binSize * (tt[0][-1] - tt[0][0]))
    #
    #
    # tt = np.where(hy >= hy.max() * 0.5)
    # if hx[tt].size > 1:
    #     binSize = y[1] - y[0]
    #     print("FWHM Y: ",binSize * (tt[0][-1] - tt[0][0]))

    # supress ordinates labels ans ticks
    axHistx.get_yaxis().set_visible(False)
    axHisty.get_xaxis().set_visible(False)
    axHistx.set_ylim(ymin=0, ymax=hy.max()*1.1)
    axHisty.set_xlim(xmin=0, xmax=hx.max() * 1.1)
    # supress abscissas labels (keep ticks)
    for tl in axHistx.get_xticklabels(): tl.set_visible(False)
    for tl in axHisty.get_yticklabels(): tl.set_visible(False)

    if title != "":
        axHistx.set_title(title)

    if add_colorbar:
        plt.colorbar(axs)

    if show:
        plt.show()

    return figure, axScatter, axHistx, axHisty



def plot(*positional_parameters, title="", xtitle="", ytitle="",
         xrange=None, yrange=None, show=1, legend=None, legend_position=None, color=None, marker=None, linestyle=None,
         xlog=False, ylog=False, figsize=None):
    """
    Makes a plot of a dataset (XY plot).

    Parameters
    ----------
    *positional_parameters : numpy arrays
        x1, y1 [, x2, y2, x3, y3, ...]
    title : str, optional
        The main title.
    xtitle : str, optional
        The X title.
    ytitle : str, optional
        The Y title.
    xrange : 2D tuple or list
        [min, max] values.
    yrange : 2D tuple or list
        [min, max] values.
    show : int, optional
        if 1 executes show_plot()
    legend : str or list
        The legend for the different datasets.
    legend_position : list or tuple
        2D list with coordinates of the legend.
    color : str
        The matplotlib color code.
    marker : str or list
        The matplotlib marker or symbol.
    linestyle : str or list
        The matplotlib linestyle code.
    xlog : boolean, optional
        Set to True for logarithmic plot axis.
    ylog : boolean, optional
        Set to True for logarithmic plot axis.
    figsize : tuple
        The matplotlib figure size.

    Returns
    -------
    tuple
        The matplotmib (fig, ax).

    """

    if isinstance(positional_parameters,tuple):
        if len(positional_parameters) == 1: # in the cvase that input is a tuple with all curves
            positional_parameters = positional_parameters[0]

    n_arguments = len(positional_parameters)
    if n_arguments == 0:
        return

    fig = plt.figure(figsize=figsize)
    if n_arguments == 1:
        y = positional_parameters[0]
        x = np.arange(y.size)
        if linestyle == None:
            linestyle = '-'
        plt.plot(x,y,label=legend,marker=marker,color=color,linestyle=linestyle)
    elif n_arguments == 2:
        x = positional_parameters[0]
        y = positional_parameters[1]
        if linestyle == None:
            linestyle = '-'
        plt.plot(x, y, label=legend, color=color, marker=marker, linestyle=linestyle)
    elif n_arguments % 2 == 0:
        if legend is None:
            legend = [None] * (n_arguments // 2)
        if color is None:
            color = [None] * (n_arguments // 2)

        if marker is None:
            marker = [None] * (n_arguments // 2)

        if linestyle is None:
            linestyle = ['-'] * (n_arguments // 2)

        for i in range(n_arguments // 2):
            plt.plot(positional_parameters[2*i],positional_parameters[2*i+1],label=legend[i],marker=marker[i],linestyle=linestyle[i],color=color[i])
    else:
        raise Exception("Incorrect number of arguments: found an odd number of data sets")
        # x = positional_parameters[0]
        # y = positional_parameters[1]
        # plt.plot(x,y,label=legend)

    ax = plt.subplot(111)
    if legend is not None:
        ax.legend(bbox_to_anchor=legend_position)

    if xlog:
        ax.set_xscale("log")

    if ylog:
        ax.set_yscale("log")

    plt.xlim( xrange )
    plt.ylim( yrange )

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)


    if show:
        plt.show()

    return fig, ax

def plot_table(*positional_parameters, errorbars=None, xrange=None, yrange=None,
               title="", xtitle="", ytitle="", show=1,
               legend=None, legend_position=None, color=None,
               xlog=False, ylog=False, figsize=None):
    """
    Makes a plot with data in tabular arrays using one array with abscissas X (Nx values), and another array
    (N, Nx) with ordinates Y for N datasets.

    Parameters
    ----------

    *positional_parameters : numpy arrays
        [X, ] Y
    errorbars : numpy array
        The values for the error bars.
    title : str, optional
        The main title.
    xtitle : str, optional
        The X title.
    ytitle : str, optional
        The Y title.
    xrange : 2D tuple or list
        [min, max] values.
    yrange : 2D tuple or list
        [min, max] values.
    show : int, optional
        if 1 executes show_plot()
    legend : str or list
        The legend for the different datasets.
    legend_position : list or tuple
        2D list with coordinates of the legend.
    color : str
        The matplotlib color code.
    xlog : boolean, optional
        Set to True for logarithmic plot axis.
    ylog : boolean, optional
        Set to True for logarithmic plot axis.
    figsize : tuple
        The matplotlib figure size.

    Returns
    -------
    tuple
       The matplotlib objects (fig, ax).

    """

    n_arguments = len(positional_parameters)
    if n_arguments == 0:
        return

    fig = plt.figure(figsize=figsize)

    if n_arguments == 1:
        y = positional_parameters[0]
        x = np.arange(y.size)
        plt.plot(x,y,label=legend)
    elif n_arguments == 2:
        x = positional_parameters[0]
        y = positional_parameters[1]

        if len(y.shape) == 1:
            y = np.reshape(y,(1,y.size))
            if isinstance(legend,str):
                legend = [legend]
            if isinstance(color,str):
                color = [color]

        for i in range(y.shape[0]):
            if legend is None:
                ilegend = None
            else:
                ilegend = legend[i]

            if color is None:
                icolor = None
            else:
                icolor = color[i]

            if errorbars is None:
                plt.plot(x,y[i],label=ilegend,color=icolor)
            else:
                plt.errorbar(x,y[i],yerr=errorbars[i],label=ilegend,color=icolor)
    else:
        raise Exception("Incorrect number of arguments")

    ax = plt.subplot(111)

    if xlog:
        ax.set_xscale("log")

    if ylog:
        ax.set_yscale("log")

    if legend is not None:
        if legend_position is None:
            legend_position = (1.1, 1.05)
        ax.legend(bbox_to_anchor=legend_position)

    plt.xlim( xrange )
    plt.ylim( yrange )

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)


    if show:
        plt.show()

    return fig, ax

def four_plots(x1, y1, x2, y2, x3, y3, x4, y4, title="", xtitle="", ytitle="", xrange=None, yrange=None, show=True):
    """
    Creates four plots in a window.

    Parameters
    ----------
    x1 : numpy array
        abscissas for plot 1.
    y1 : numpy array
        ordinates for plot 1.
    x2 : numpy array
        abscissas for plot 2.
    y2 : numpy array
        ordinates for plot 2.
    x3 : numpy array
        abscissas for plot 3.
    y3 : numpy array
        ordinates for plot 3.
    x4 : numpy array
        abscissas for plot 4.
    y4 : numpy array
        ordinates for plot 4.
    title : str or list, optional
        a string or list of 4 strings with title.
    xtitle : str or list, optional
        a string or list of 4 strings with title for X.
    ytitle : str or list, optional
        a string or list of 4 strings with title for Y.
    xrange : tuple or list, optional
        the X range for all plots.
    yrange : tuple or list, optional
        the Y range for all plots.
    show : boolean, optional
        if True, execute plot_show() at the end.

    Returns
    -------
    tuple
        The matplotlib objects (f, ax00, ax01, ax10, ax11).

    """

    if isinstance(title,list):
        Title = title
    else:
        Title = [title,title,title,title]

    if isinstance(xtitle,list):
        Xtitle = xtitle
    else:
        Xtitle = [xtitle,xtitle,xtitle,xtitle]

    if isinstance(ytitle,list):
        Ytitle = ytitle
    else:
        Ytitle = [ytitle,ytitle,ytitle,ytitle]

    # Create subplots.
    f, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2, sharex="all", sharey="all")

    ax00.plot(x1,y1, "-")
    ax00.set_title(  Title[0])
    ax00.set_xlabel(Xtitle[0])
    ax00.set_ylabel(Ytitle[0])
    ax00.set_xlim(xrange)
    ax00.set_ylim(yrange)


    ax01.plot(x2,y2, "-")
    ax01.set_title(  Title[1])
    ax01.set_xlabel(Xtitle[1])
    ax01.set_ylabel(Ytitle[1])
    ax01.set_xlim(xrange)
    ax01.set_ylim(yrange)


    ax10.plot(x3,y3, "-")
    ax10.set_title(  Title[2])
    ax10.set_xlabel(Xtitle[2])
    ax10.set_ylabel(Ytitle[2])
    ax10.set_xlim(xrange)
    ax10.set_ylim(yrange)


    ax11.plot(x4,y4, "-")
    ax11.set_title(  Title[3])
    ax11.set_xlabel(Xtitle[3])
    ax11.set_ylabel(Ytitle[3])
    ax11.set_xlim(xrange)
    ax11.set_ylim(yrange)

    if show: plt.show()

    return f, ax00, ax01, ax10, ax11

def plot_surface(mymode, theta, psi, title="TITLE", xtitle="", ytitle="", ztitle="", legend=None, cmap=None,
                 figsize=None,show=1):
    """
    Plots a 2D surface given by an array mymode(theta, phi).

    Parameters
    ----------
    mymode : numpy array
        The 2D array with the surface to plot.
    theta : numpy array
        The array with X (H) values.
    psi : numpy array
        The array with Y (V) values.
    title : str, optional
        The main title.
    xtitle : str, optional
        The X title.
    ytitle : str, optional
        The Y title.
    ztitle : str, optional
        The main title.
    legend : str or list
        The legend for the different datasets.
    cmap : str, optional
        The matplotlib color map.
    figsize : tuple
        The matplotlib figure size.
    show : boolean, optional
        If True, executes plot_show() at the end.

    Returns
    -------
    tuple
        The matplotlib objects (fig, ax).

    """

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D

    ftheta, fpsi = np.meshgrid(theta, psi)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')

    II0 = mymode.T

    if cmap == None:
        cmap = cm.coolwarm

    print(II0.shape,ftheta.shape,fpsi.shape)
    surf = ax.plot_surface(ftheta, fpsi, II0, rstride=1, cstride=1, cmap=cmap,
                           linewidth=0, antialiased=False)

    ax.set_zlim(II0.min(),II0.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_zlabel(ztitle)

    if show:
        plt.show()

    return fig, ax

def plot_scatter(x, y, show=1, nbins=100, xrange=None, yrange=None, plot_histograms=True, title="", xtitle="", ytitle=""):
    """
    makes a scatter plot with histograms.

    Parameters
    ----------
    x : numpy array
        x data array.
    y : numpy array
        y data array.
    show : boolean, optional
        If True, executes plot_show() at the end.
    nbins : int
        The number of bins for the histograms.
    xrange : tuple or list, optional
        the X range for all plots.
    yrange : tuple or list, optional
        the Y range for all plots.
    plot_histograms : boolean, optional
        Flag for plotting histograms: False or 0: plot no histograms;
            True or 1: plot both histograms;
            2: plot histograms vs abscissas only;
            3: plot histogram vs ordinates only.
    title : str, optional
        The main title.
    xtitle : str, optional
        The X title.
    ytitle : str, optional
        The Y title.

    Returns
    -------
    tuple
        The matplotlib objects (fig, axScatter).

    """
    from matplotlib.ticker import NullFormatter

    # the random data

    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes

    if plot_histograms:
        left, width    = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.02
        rect_scatter = [left, bottom, width, height]
        rect_histx   = [left, bottom_h, width, 0.2]
        rect_histy   = [left_h, bottom, 0.2, height]
    else:
        left, width    = 0.1, 0.8
        bottom, height = 0.1, 0.8
        rect_scatter = [left, bottom, width, height]

    # start with a rectangular Figure
    fig = plt.figure(figsize=(8,8))

    axScatter = plt.axes(rect_scatter)
    if plot_histograms:
        if plot_histograms == 2:
            axHistx = plt.axes(rect_histx)
            axHistx.xaxis.set_major_formatter(nullfmt)
        elif plot_histograms == 3:
            axHisty = plt.axes(rect_histy)
            axHisty.yaxis.set_major_formatter(nullfmt)
        else:
            axHistx = plt.axes(rect_histx)
            axHisty = plt.axes(rect_histy)
            # no labels
            axHistx.xaxis.set_major_formatter(nullfmt)
            axHisty.yaxis.set_major_formatter(nullfmt)

    # now determine nice limits by hand:
    binwidth = np.array([x.max() - x.min(), y.max() - y.min()]).max() / nbins

    if xrange == None:
        xrange = np.array((x.min(), x.max()))
    if yrange == None:
        yrange = np.array((y.min(), y.max()))

    # the scatter plot:
    axScatter.scatter(x, y, marker='.', edgecolor='b', s=.1)

    axScatter.set_xlabel(xtitle)
    axScatter.set_ylabel(ytitle)
    if not plot_histograms:
        axScatter.set_title(title)

    axScatter.set_xlim( xrange )
    axScatter.set_ylim( yrange )

    if plot_histograms:
        bins_x = np.arange(xrange[0], xrange[1] + binwidth, binwidth)
        if plot_histograms == 2:
            axHistx.hist(x, bins=nbins)
            axHistx.set_xlim( axScatter.get_xlim() )
            axHistx.set_title(title)
        elif plot_histograms == 3:
            axHisty.hist(y, bins=nbins, orientation='horizontal')
            axHisty.set_ylim( axScatter.get_ylim() )
        else:
            axHistx.hist(x, bins=nbins, range=xrange)
            axHisty.hist(y, bins=nbins, range=yrange, orientation='horizontal')

            axHistx.set_xlim( axScatter.get_xlim() )

            axHistx.set_title(title)
            axHisty.set_ylim( axScatter.get_ylim() )


    if show: plt.show()

    if plot_histograms:
        if plot_histograms == 2:
            return fig,axScatter,axHistx
        elif plot_histograms == 3:
            return fig,axScatter,axHisty
        else:
            return fig,axScatter,axHistx,axHisty
    else:
        return fig, axScatter

def plot_contour(z, x, y, title="TITLE", xtitle="", ytitle="", xrange=None, yrange=None, plot_points=0, contour_levels=20,
                 cmap=None, cbar=True, fill=False, cbar_title="", figsize=None, show=1):
    """
    Plot a contour curves plot from data z(x,y).

    Parameters
    ----------
    z : numpy array
        The 2D data to plot.
    x : numpy array
        The 1D data for X axis.
    y : numpy array
        The 1D data for Y axis.
    title : str, optional
        The main title.
    xtitle : str, optional
        The X title.
    ytitle : str, optional
        The Y title.
    xrange : tuple or list, optional
        the X range for all plots.
    yrange : tuple or list, optional
        the Y range for all plots.
    plot_points : int, optional
        The number of ppoints for the contours.
    contour_levels : int, optional
        The number of contour levels.
    cmap : str, optional
        The matplotlib color map.
    cbar : boolean, optional
        Flag to plot the color table.
    fill : boolean, optional
        Flag to fill the contours.
    cbar_title : str, optional
        The titles for the contours.
    figsize : tuple
        The matplotlib figure size.
    show : boolean, optional
        If True, executes plot_show() at the end.

    Returns
    -------
    instance of plt.contourf
        The matplotlib plt.contourf instance.

    """

    fig = plt.figure(figsize=figsize)

    if fill:
        fig = plt.contourf(x, y, z.T, contour_levels, cmap=cmap, origin='lower')
    else:
        fig = plt.contour( x, y, z.T, contour_levels, cmap=cmap, origin='lower')

    if cbar:
        cbar = plt.colorbar(fig)
        cbar.ax.set_ylabel(cbar_title)


    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    # the scatter plot:
    if plot_points:
        axScatter = plt.subplot(111)
        axScatter.scatter( np.outer(x,np.ones_like(y)), np.outer(np.ones_like(x),y))

    # set axes range
    plt.xlim(xrange)
    plt.ylim(yrange)

    if show:
        plt.show()

    return fig

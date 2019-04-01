"""


GOL: Graphics in One Line

To make matplotlib plots in a single command line

(I am tired of polluting my code with matplotlib instructions)


"""

__author__ = "Manuel Sanchez del Rio"
__contact__ = "srio@esrf.eu"
__copyright = "ESRF, 2016"

import numpy as np
try:
    import matplotlib.pylab as plt
except:
    raise ImportError("Please install matplotlib to allow graphics")

# try:
#     plt.switch_backend("Qt5Agg")
# except:
#     raise Exception("Failed to set matplotlib backend to Qt5Agg")

def plot_show():
    plt.show()

def plot_image(*positional_parameters,title="TITLE",xtitle=r"X",ytitle=r"Y",
               xrange=None, yrange=None,
               cmap=None,aspect=None,show=1,
               add_colorbar=True,figsize=None):

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

    return fig,ax


def plot_image_with_histograms(*positional_parameters,
            title="",xtitle=r"X",ytitle=r"Y",
            xrange=None, yrange=None,
            cmap=None,aspect_ratio=None,show=True,
            add_colorbar=False,figsize=(8,8)
            ):

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


    axScatter.axis(xmin=hfactor*xrange[0],xmax=xrange[1])
    axScatter.axis(ymin=vfactor*yrange[0],ymax=yrange[1])

    if aspect_ratio is not None:
        axScatter.set_aspect(aspect_ratio)

    axs = axScatter.pcolormesh(x,y,z,cmap=cmap)

    #
    #histograms
    #
    axHistx = figure.add_axes(rect_histx, sharex=axScatter)
    axHisty = figure.add_axes(rect_histy, sharey=axScatter)

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

    # supress abscissas labels (keep ticks)
    for tl in axHistx.get_xticklabels(): tl.set_visible(False)
    for tl in axHisty.get_yticklabels(): tl.set_visible(False)

    if title != "":
        axHistx.set_title(title)

    if add_colorbar:
        plt.colorbar(axs)

    if show:
        plt.show()

    return figure #,ax



def plot(*positional_parameters,title="",xtitle="",ytitle="",
         xrange=None,yrange=None,show=1,legend=None,legend_position=None,color=None,marker=None,linestyle=None,
         xlog=False,ylog=False,figsize=None):

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
        plt.plot(x,y,label=legend,color=color,marker=marker,linestyle=linestyle)
    elif n_arguments == 4:
        x1 = positional_parameters[0]
        y1 = positional_parameters[1]
        x2 = positional_parameters[2]
        y2 = positional_parameters[3]
        if legend is None:
            legend1 = None
            legend2 = None
        else:
            legend1 = legend[0]
            legend2 = legend[1]

        if color is None:
            color1 = None
            color2 = None
        else:
            color1 = color[0]
            color2 = color[1]

        if marker is None:
            marker1 = None
            marker2 = None
        else:
            marker1 = marker[0]
            marker2 = marker[1]

        if linestyle is None:
            linestyle1 = '-'
            linestyle2 = '-'
        else:
            linestyle1 = linestyle[0]
            linestyle2 = linestyle[1]

        plt.plot(x1,y1,label=legend1,marker=marker1,linestyle=linestyle1,color=color1)
        plt.plot(x2,y2,label=legend2,marker=marker2,linestyle=linestyle2,color=color2)
    elif n_arguments == 6:
        x1 = positional_parameters[0]
        y1 = positional_parameters[1]
        x2 = positional_parameters[2]
        y2 = positional_parameters[3]
        x3 = positional_parameters[4]
        y3 = positional_parameters[5]
        if legend is None:
            legend1 = None
            legend2 = None
            legend3 = None
        else:
            legend1 = legend[0]
            legend2 = legend[1]
            legend3 = legend[2]

        if color is None:
            color1 = None
            color2 = None
            color3 = None
        else:
            color1 = color[0]
            color2 = color[1]
            color3 = color[2]

        if marker is None:
            marker1 = None
            marker2 = None
            marker3 = None
        else:
            marker1 = marker[0]
            marker2 = marker[1]
            marker3 = marker[2]

        if linestyle is None:
            linestyle1 = '-'
            linestyle2 = '-'
            linestyle3 = '-'
        else:
            linestyle1 = linestyle[0]
            linestyle2 = linestyle[1]
            linestyle3 = linestyle[2]

        plt.plot(x1,y1,label=legend1,marker=marker1,linestyle=linestyle1,color=color1)
        plt.plot(x2,y2,label=legend2,marker=marker2,linestyle=linestyle2,color=color2)
        plt.plot(x3,y3,label=legend3,marker=marker3,linestyle=linestyle3,color=color3)
    elif n_arguments == 8:
        x1 = positional_parameters[0]
        y1 = positional_parameters[1]
        x2 = positional_parameters[2]
        y2 = positional_parameters[3]
        x3 = positional_parameters[4]
        y3 = positional_parameters[5]
        x4 = positional_parameters[6]
        y4 = positional_parameters[7]
        if legend is None:
            legend1 = None
            legend2 = None
            legend3 = None
            legend4 = None
        else:
            legend1 = legend[0]
            legend2 = legend[1]
            legend3 = legend[2]
            legend4 = legend[3]
        if color is None:
            color1 = None
            color2 = None
            color3 = None
            color4 = None
        else:
            color1 = color[0]
            color2 = color[1]
            color3 = color[2]
            color4 = color[3]

        if marker is None:
            marker1 = None
            marker2 = None
            marker3 = None
            marker4 = None
        else:
            marker1 = marker[0]
            marker2 = marker[1]
            marker3 = marker[2]
            marker4 = marker[3]

        if linestyle is None:
            linestyle1 = '-'
            linestyle2 = '-'
            linestyle3 = '-'
            linestyle4 = '-'
        else:
            linestyle1 = linestyle[0]
            linestyle2 = linestyle[1]
            linestyle3 = linestyle[2]
            linestyle4 = linestyle[3]

        plt.plot(x1,y1,label=legend1,marker=marker1,linestyle=linestyle1,color=color1)
        plt.plot(x2,y2,label=legend2,marker=marker2,linestyle=linestyle2,color=color2)
        plt.plot(x3,y3,label=legend3,marker=marker3,linestyle=linestyle3,color=color3)
        plt.plot(x4,y4,label=legend4,marker=marker4,linestyle=linestyle4,color=color4)
    else:
        raise Exception("Incorrect number of arguments, maximum 4 data sets")
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

    return fig,ax

def plot_table(*positional_parameters,errorbars=None,xrange=None,yrange=None,
               title="",xtitle="",ytitle="",show=1,
               legend=None,legend_position=None,color=None,
               xlog=False,ylog=False,figsize=None):

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

    return fig,ax
def four_plots(x1,y1,x2,y2,x3,y3,x4,y4,title="",xtitle="",ytitle="",xrange=None,yrange=None,show=True):
    """
    Creates four plots in a window

    :param x1: abscissas for plot 1
    :param y1: ordinates for plot 1
    :param x2: abscissas for plot 2
    :param y2: ordinates for plot 2
    :param x3: abscissas for plot 3
    :param y3: ordinates for plot 3
    :param x4: abscissas for plot 4
    :param y4: ordinates for plot 4
    :param title: a string or list of 4 strings with title
    :param xtitle: a string or list of 4 strings with title for X
    :param ytitle: a string or list of 4 strings with title for Y
    :param xrange: the X range for all plots
    :param yrange: the Y range for all plots
    :param show:
    :return:
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

    return f,ax00,ax01,ax10,ax11

def plot_surface(mymode,theta,psi,title="TITLE",xtitle="",ytitle="",ztitle="",legend=None,cmap=None,
                 figsize=None,show=1):

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

    return fig,ax

def plot_scatter(x,y,show=1,nbins=100,xrange=None,yrange=None,plot_histograms=True,title="",xtitle="",ytitle=""):
    """

    makes a scatter plot with histograms

    :param x: x data array
    :param y: y data arrayif False the plot is not shown (use  not show
    :param show: if False the plot is not shown (use  plot_show() later on)
    :param nbins: number of bins for plots
    :param xrange: [xmin,xmax] range for abscissas
    :param yrange: [ymin,ymax] range for ordinates
    :param plot_histograms: Flag to plot:
            False or 0: plot no histograms
            True or 1: plot both histograms
            2: plot histograms vs abscissas only
            3: plot histogram vs ordinates only
    :param title: string with a title
    :param xtitle: string with abscissas label
    :param ytitle: string with ordinates label
    :return: the matplotlib objects with the elements created
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
            axHistx.hist(x, bins=nbins)
            axHisty.hist(y, bins=nbins, orientation='horizontal')

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
        return fig,axScatter

def plot_contour(z,x,y,title="TITLE",xtitle="",ytitle="",xrange=None,yrange=None,plot_points=0,contour_levels=20,
                 cmap=None,cbar=True,fill=False,cbar_title="",figsize=None,show=1):

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
#
# examples
#

def example_plot_image():
    x = np.linspace(-4, 4, 90)
    y = np.linspace(-4, 4, 90)
    print('Size %d pixels' % (len(x) * len(y)))
    z = np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)
    plot_image(z,x,y,title="example_plot_image",xtitle=r"X [$\mu m$]",ytitle=r"Y [$\mu m$]",cmap=None,show=1)

def example_plot_image_with_histograms():
    x = np.linspace(-4, 4, 200)
    y = np.linspace(-4, 4, 90)
    print('Size %d pixels' % (len(x) * len(y)))
    z = -np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)
    plot_image_with_histograms(z,x,y,title="example_plot_image",xtitle=r"X [$\mu m$]",ytitle=r"Y [$\mu m$]",
                               cmap=None,show=1,figsize=(8,8),add_colorbar=True)

def example_plot_surface():
    x = np.linspace(-4, 4, 20)
    y = np.linspace(-4, 4, 20)
    print('Size %d pixels' % (len(x) * len(y)))
    z = np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)
    plot_surface(z,x,y,title="example_plot_surface",xtitle=r"X [$\mu m$]",ytitle=r"Y [$\mu m$]",cmap=None,show=1)

def example_plot_scatter():
    #example motivated by http://www.ster.kuleuven.be/~pieterd/python/html/core/scipystats.html
    from scipy import stats
    # x = np.random.rand(1000)
    # y = np.random.rand(1000)
    x = stats.norm.rvs(size=2000)
    y = stats.norm.rvs(scale=0.5, size=2000)
    data = np.vstack([x+y, x-y])
    f = plot_scatter(data[0],data[1],xrange=[-10,10],title="example_plot_scatter",
                     xtitle=r"X [$\mu m$]",ytitle=r"Y [$\mu m$]",plot_histograms=2,show=0)
    f[1].plot(data[0],data[0]) # use directly matplotlib to overplot
    plot_show()

def example_plot_contour():
    # inspired by http://stackoverflow.com/questions/10291221/axis-limits-for-scatter-plot-not-holding-in-matplotlib
    # random data
    x = np.random.randn(50)
    y = np.random.randn(100)

    X, Y = np.meshgrid(y, x)
    Z1 = plt.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = plt.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    Z = 10 * (Z1 - Z2)

    plot_contour(Z,x,y,title='example_plot_contour',xtitle='x-stuff',ytitle='y-stuff',plot_points=1,show=1)

def example_plot_one_curve():
    x = np.linspace(-100,100,10)
    y = x**2
    plot(x,y,xtitle=r'$x$',title="example_plot_one_curve",
         ytitle=r'$y=f(x)=x^2$',legend="Example 1",color='pink',marker='o',linestyle=None,
         figsize=(4,8),show=1)

def example_plot_one_curve_log():
    x = np.linspace(-100,100,10)
    y = x**2
    plot(x,y,xtitle=r'$x$',title="example_plot_one_curve",
         ytitle=r'$y=f(x)=x^2$',legend="Example 1",color='pink',marker='o',linestyle=None,xlog=1,ylog=1,show=1)

def example_plot_two_curves():
    x1 = np.linspace(-100,100,1000)
    y1 = x1**2
    x2 = np.linspace(0,200,700)
    y2 = x2**2.1
    plot(x1,y1,x2,y2,xtitle=r'$x$',title="example_plot_two_curves",
         ytitle=r'$y=f(x)$',legend=[r"$x^2$",r"$x^{2.1}$"],color=['green','blue'],marker=[' ','o'],linestyle=['-',' '],show=1)

def example_plot_table():
    x1 = np.linspace(0,100,100)
    out = np.zeros((6,x1.size))
    out[0,:] = x1**2
    out[1,:] = x1**2.1
    out[2,:] = x1**2.2
    out[3,:] = x1**2.3
    out[4,:] = x1**2.4
    out[5,:] = x1**2.5
    # another way
    # out = np.vstack( (
    #     x1**2,
    #     x1**2.1,
    #     x1**2.2,
    #     x1**2.3,
    #     x1**2.4,
    #     x1**2.5 ))
    legend=np.arange(out.shape[0]).astype("str")
    plot_table(x1,out,xtitle=r'$x$',ytitle=r'$y=f(x)$',title="example_plot_table",legend=legend,show=1)

def example_plot_table_one_curve():
    x1 = np.linspace(-100,100,1000)
    out = x1**2
    plot_table(x1,out,title="example_plot_table_one_curve",xtitle=r'$x$',ytitle=r'$y=f(x)$',legend="Example 1",color='pink',show=1)

def example_plot_table_with_errorbars():
    x = np.linspace(0,100,30)
    out = np.zeros((2,x.size))
    out[0,:] = 1e-3 * x**2
    out[1,:] = 5 + 1e-3 * x**2
    yerr = np.sqrt(out)
    yerr[1,:] = 1.0
    plot_table(x,out,errorbars=yerr,title="example_plot_table_with_errorbars",xtitle=r'$x$',ytitle=r'$y=f(x)=x^2$',xrange=[20,80],
               legend=["Statistical error","Constant error"],color=['black','magenta'],show=1)

def example_plot_image_ascent():
    from scipy.misc import ascent

    ascent = np.rot90(ascent(),-1)
    plot_image(ascent,np.arange(0,ascent.shape[0]),np.arange(0,ascent.shape[1]),cmap='gray' )
#
# main
#
if __name__ == "__main__":
    pass
    # example_plot_one_curve()
    # example_plot_two_curves()
    # example_plot_one_curve_log()
    # example_plot_table()
    # example_plot_table_one_curve()
    # example_plot_table_with_errorbars()
    # example_plot_image()
    # example_plot_image_with_histograms()
    # example_plot_surface()
    # example_plot_contour()
    # example_plot_scatter()
    # example_plot_image_ascent()

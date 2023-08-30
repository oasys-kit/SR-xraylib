"""
This class is intended to be used for writing data of the form y=f(x) (1D dataset or simply "dataset")
or z=f(x,y) (2D dataset or "image") in an HDF5 file. It includes the NX attributes necessary for automatic
plotting using "silx view".

Note:
    The image arrays we use here have: x=horizontal=axis0 and  y=vertical=axis1.
    However, the hdf5 file uses the C / python convention so: x=horizontal=axis1 and  y=vertical=axis0.
    To allow display graphics correctly, the array with the image stored in the hdf5 file is transposed.
    This has no effect for the user that use "silx view" for display, but must be taken into account when retrieving the data from the file.

srio@esrf.eu 2018-03-23

"""
import h5py,sys,time, os


class H5SimpleWriter(object):
    """
    Constructor.

    Parameters
    ----------
    filename : str
        File name to be written.
    creator : str
        A name to indicate the creator or author (for info purposes).

    """
    def __init__(self, filename, creator):
        self.creator = creator
        self.filename = filename

        # for internal use, modify at your own risk...
        self.label_image_data   = "image_data"
        self.label_image_axis_x = b'axis_x'
        self.label_image_axis_y = b'axis_y'

        self.label_dataset_y = b'y'
        self.label_dataset_x = b'x'

        self.label_stack_data   = "stack_data"
        self.label_stack_axis0 = b'axis0'
        self.label_stack_axis1 = b'axis1'
        self.label_stack_axis2 = b'axis2'

    @classmethod
    def initialize_file(cls, filename, creator="H5BasicWriter", overwrite=True):
        """
        Creates an h5 file and initializes it.

        Parameters
        ----------
        filename : str
            Fine name.
        creator : str, optional
            a name of the creator or author.
        overwrite : boolean, optional
            If True, overwrites the file if exists. If False, if the file exists it will raise an error.

        """
        if overwrite:
            try:
                os.remove(filename)
            except:
                pass
        tmp = H5SimpleWriter(filename, creator)
        tmp.create_new_file()
        tmp.add_file_header()
        return tmp

    def set_label_image(self, str_data, str_x, str_y):
        """
        Sets the labels for 2D data.

        Parameters
        ----------
        str_data : str
            The label for the data.
        str_x : str
            The label for the X (horizontal).
        str_y : str
            The label for the Y (vertical).

        """
        self.label_image_data   = str_data
        self.label_image_axis_x = str_x
        self.label_image_axis_y = str_y

    def set_label_dataset(self, str_x, str_y):
        """
        Sets the labels for 1D data.

        Parameters
        ----------
        str_x : str
            The label for the X (horizontal).
        str_y : str
            The label for the Y (vertical).

        Returns
        -------

        """
        self.label_dataset_x = str_x
        self.label_dataset_y = str_y

    def set_label_stack(self, str_data, str_0, str_1, str_2):
        """
        Sets the labels for 3D data.

        Parameters
        ----------
        str_data : str
            The label for data.
        str_0 : str
            The label for X.
        str_1 : str
            The label for Y.
        str_2 : str
            The label for Z.

        Returns
        -------

        """
        self.label_stack_data   = str_data
        self.label_stack_axis0 = str_0
        self.label_stack_axis1 = str_1
        self.label_stack_axis2 = str_2

    def create_new_file(self):
        """
        Creates a new (empty) file.

        Returns
        -------
        boolean
            success (True) or error (False)

        """
        try:
            f = h5py.File(self.filename, 'w')
            f.close()
            return True
        except:
            return False


    def add_file_header(self):
        """
        Adds a "standard" file header.

        Returns
        -------
        boolean
            success (True) or error (False)

        """
        try:
            sys.stdout.flush()
            f = h5py.File(self.filename, 'a')
            # point to the default data to be plotted
            f.attrs['default']          = 'entry'
            # give the HDF5 root some more attributes
            f.attrs['file_name']        = self.filename
            f.attrs['file_time']        = time.time()
            f.attrs['creator']          = self.creator
            f.attrs['HDF5_Version']     = h5py.version.hdf5_version
            f.attrs['h5py_version']     = h5py.version.version
            f.close()
            return True
        except:
            return False

    def create_entry(self, entry_name, root_entry=None, nx_default=None):
        """
        Creates a HDF5 group (or NX "entry") which is in fact the "folder" in the file that will contain the datasets.

        Parameters
        ----------
        entry_name : str
            string with the name of the entry.
        root_entry : str, optional
            the root entry name (default: None)
        nx_default : str, optional
            the name of the dataset to inside this entry (to be added later) with the default plot
            (i.e.,when clicking the entry name).

        """
        f = h5py.File(self.filename, 'a')

        if root_entry is None:
            f1 = f.create_group(entry_name)
        else:
            f0 = f[root_entry]
            f1 = f0.create_group(entry_name)



        if nx_default is not None:
            f1.attrs['NX_class'] = 'NXentry'
            f1.attrs['default'] = nx_default
        f.close()

    def add_key(self, key, value, entry_name=None):
        """
        Adds a pair (key, value).

        Parameters
        ----------
        key : str
            The key name.
        value : int, float or numpy array.
            The value.
        entry_name : str, optional
            The entry_name to which the pair (key, value) will be added. Default: at main level.

        """
        f = h5py.File(self.filename, 'a')

        if entry_name is None:
            f[key] = value
        else:
            f1 = f[entry_name]
            f1[key] = value

        f.close()

    def add_dataset(self, x, y, dataset_name="tmp", entry_name=None, title_x="", title_y=""):
        """
        Adds a dataset (1D data) to the file.

        Parameters
        ----------
        x : numpy array
            The dataset abscissas.
        y : numpy array
            The dataset ordinates.
        dataset_name : str, optional
            A name for the data set.
        entry_name : str, optional
            The entry name where the dataset is added.
        title_x : str, optional
            The dataset X title.
        title_y : str, optional
            The dataset Y title.

        """
        f = h5py.File(self.filename, 'a')

        if entry_name is None:
            f1 = f
        else:
            f1 = f[entry_name]

        # Add NX plot attribites for automatic plot with silx view

        f2 = f1.create_group(dataset_name)
        f2.attrs['NX_class'] = 'NXdata'
        f2.attrs['signal'] = self.label_dataset_y # b'y'
        f2.attrs['axes']   = self.label_dataset_x # b'x'

        # Y data
        ds = f2.create_dataset(self.label_dataset_y, data=y)
        ds.attrs['long_name'] = title_y    # suggested X axis plot label

        # X axis data
        ds = f2.create_dataset(self.label_dataset_x, data=x)
        ds.attrs['long_name'] = title_x    # suggested X axis plot label

        f.close()

    def add_image(self,image,image_x=None,image_y=None,image_name="myimage",entry_name=None,title_x="",title_y=""):
        """
        Adds an image (2D data) to the file.

        Parameters
        ----------
        image : numpy array
            The 2D array with the data (image) values.
        image_x : numpy array, optional
            The 1D array with the values of the X axis.
        image_y : numpy array, optional
            The 1D array with the values of the Y axis.
        image_name : str, optional
            A name for the added image data.
        entry_name : str, optional
            The name of the entry name where the data is added.
        title_x : str, optional
            A title for the X axis.
        title_y : str, optional
            A title for the Y axis.

        """

        if image_x is None:
            image_x = numpy.arange(image.shape[0])

        if image_y is None:
            image_y = numpy.arange(image.shape[1])

        f = h5py.File(self.filename, 'a')

        if entry_name is None:
            f1 = f
        else:
            f1 = f[entry_name]

        f2 = f1.create_group(image_name)

        f2.attrs['NX_class'] = 'NXdata'
        f2.attrs['signal'] = '%s'%(self.label_image_data)
        f2.attrs['axes'] = [self.label_image_axis_y, self.label_image_axis_x]

        # Image data
        ds = f2.create_dataset(self.label_image_data, data=image.T)
        ds.attrs['interpretation'] = 'image'

        # X axis data
        ds = f2.create_dataset(self.label_image_axis_y, data=image_y)
        # ds.attrs['units'] = 'microns'
        ds.attrs['long_name'] = title_y    # suggested X axis plot label

        # Y axis data
        ds = f2.create_dataset(self.label_image_axis_x, data=image_x)
        # ds.attrs['units'] = 'microns'
        ds.attrs['long_name'] = title_x    # suggested Y axis plot label

        f.close()

    def add_stack(self,e,h,v,p,stack_name="Radiation",entry_name=None,
                     title_0="",title_1="",title_2=""):
        """
        Adds a stack (3D data) to the file.

        Parameters
        ----------
        e : numpy array
            An array with values for axis 0.
        h : numpy array
            An array with values for axis 1.
        v : numpy array
            An array with values for axis 2.
        p : numpy array
            A 3D array with the stack values.
        stack_name : str, optional
            A name for the stack added.
        entry_name : str, optional
            The name of the entry where the data are added.
        title_0 : str, optional
            A title for axis 0.
        title_1 : str, optional
            A title for axis 1.
        title_2 : str, optional
            A title for axis 2.

        """

        f = h5py.File(self.filename, 'a')

        if entry_name is None:
            f1 = f
        else:
            f1 = f[entry_name]


        f2 = f1.create_group(stack_name)

        f2.attrs['NX_class'] = 'NXdata'
        f2.attrs['signal'] = '%s'%("stack_data")
        f2.attrs['axes'] = [self.label_stack_axis0, self.label_stack_axis1, self.label_stack_axis2]

        # Image data
        ds = f2.create_dataset(self.label_stack_data, data=p)

        ds = f2.create_dataset(self.label_stack_axis0, data=e)
        ds.attrs['long_name'] = title_0    # suggested 0 axis plot label

        # X axis data
        ds = f2.create_dataset(self.label_stack_axis1, data=h)
        ds.attrs['long_name'] = title_1    # suggested 1 axis plot label


        # Y axis data
        ds = f2.create_dataset(self.label_stack_axis2, data=v)
        ds.attrs['long_name'] = title_2    # suggested 2 axis plot label

        f.close()

    def add_deepstack(self,list_of_axes_arrays,stack_array,stack_name="mydeepstack",entry_name=None,
                     list_of_axes_labels=None, list_of_axes_titles=None):

        """
        Adds a stack (3D data) to the file, like add_stack, but with a different way to enter the data.

        Parameters
        ----------
        list_of_axes_arrays : list
            [x,y,z] the values of axes 0, 1 and 2.
        stack_array : numpy array
            A 3D array with the stack values.
        stack_name : str, optional
            A name for the stack added.
        entry_name : str, optional
            The name of the entry where the data are added.
        list_of_axes_labels : list
            ['label0','label1','label2]
        list_of_axes_titles : list
            ['title0','title1','title2]

        """
        f = h5py.File(self.filename, 'a')

        if entry_name is None:
            f1 = f
        else:
            f1 = f[entry_name]


        f2 = f1.create_group(stack_name)

        f2.attrs['NX_class'] = 'NXdata'
        f2.attrs['signal'] = '%s'%("stack_data")

        if list_of_axes_labels is None:
            list_of_axes_labels = []
            for i in range(len(list_of_axes_arrays)):
                list_of_axes_labels.append("axis%d" % i)
            f2.attrs['axes'] = list_of_axes_labels

        if list_of_axes_titles is None:
            list_of_axes_titles = list_of_axes_labels

        # stack data
        ds = f2.create_dataset(self.label_stack_data, data=stack_array)
        for i in range(len(list_of_axes_arrays)):
            ds = f2.create_dataset(list_of_axes_labels[i], data=list_of_axes_arrays[i])
            ds.attrs['long_name'] =  list_of_axes_titles[i]   # suggested 0 axis plot label

        f.close()

#
########################################################################################################################
#

if __name__ == "__main__":

    import numpy

    # example of data_set and image
    if True:
        #
        # Create some Gaussian image
        #
        index_x2 = 150
        index_y2 = 90

        x_coordinates = numpy.linspace(-20,20,400)
        y_coordinates = numpy.linspace(-10,10,200)

        X = numpy.outer(x_coordinates,numpy.ones_like(y_coordinates))
        Y = numpy.outer(numpy.ones_like(x_coordinates),y_coordinates)

        Z = numpy.exp( numpy.sqrt( (X-x_coordinates[index_x2])**2+(Y-y_coordinates[index_y2])**2)/2/30 )

        #
        # initialize file
        #
        h5w = H5SimpleWriter.initialize_file("test.h5",creator="h5_basic_writer.py")

        # this is optional
        h5w.set_label_image("my name for image",b'my name for image axis x',b'my name for image axis y')
        h5w.set_label_dataset(b'my name for x',b'my name for y')

        #
        # put data in file
        #

        # add some data at the main level
        h5w.add_key("image shape",Z.shape)

        for nmax in range(6):
            print("Calculating iteration %d"%nmax)

            index_x2 = numpy.min( [x_coordinates.size-1, index_x2 + int(200*(numpy.random.random()-0.5))] )
            index_y2 = numpy.min( [y_coordinates.size-1, index_y2 + int(100*(numpy.random.random()-0.5))] )

            print(index_x2,index_y2)

            Z = numpy.exp( numpy.sqrt( (X-x_coordinates[index_x2])**2+(Y-y_coordinates[index_y2])**2)/2/30 )


            # create the entry for this iteration and set default plot to "Wintensity"
            h5w.create_entry("iteration%d"%nmax,nx_default="Wintensity")

            # add some data for info at this entry level
            h5w.add_key("r2_indices",[index_x2,index_y2], entry_name="iteration%d"%nmax)
            h5w.add_key("r2",[x_coordinates[index_x2],y_coordinates[index_y2]], entry_name="iteration%d"%nmax)

            # add the images at this entry level
            h5w.add_image(Z,1e3*x_coordinates,1e3*y_coordinates,
                         entry_name="iteration%d"%nmax,image_name="Wamplitude",
                         title_x="X [mm]",title_y="Y [mm]")

            h5w.add_image(numpy.absolute(Z),1e3*x_coordinates,1e3*y_coordinates,
                        entry_name="iteration%d"%nmax,image_name="Wintensity",
                        title_x="X [mm]",title_y="Y [mm]")

                    # add that y=f(x) data at this entry level
            h5w.add_dataset(1e3*x_coordinates,Z[:,int(y_coordinates.size/2)],
                        entry_name="iteration%d"%nmax,dataset_name="profileH",
                        title_x="X [mm]",title_y="Profile along X")

            h5w.add_dataset(1e3*y_coordinates,Z[int(x_coordinates.size/2),:],
                        entry_name="iteration%d"%nmax,dataset_name="profileV",
                        title_x="Y [mm]",title_y="Profile along Y")

    # example of stack
    if True:
        #
        # Create some Gaussian images
        #
        x_coordinates = numpy.linspace(-20, 20, 400)
        y_coordinates = numpy.linspace(-10, 10, 200)

        X = numpy.outer(x_coordinates, numpy.ones_like(y_coordinates))
        Y = numpy.outer(numpy.ones_like(x_coordinates), y_coordinates)

        index_x2 = 150
        index_y2 = 90
        nmax = 6
        for i in range(nmax):
            index_x2 = numpy.min([x_coordinates.size - 1, index_x2 + int(200 * (numpy.random.random() - 0.5))])
            index_y2 = numpy.min([y_coordinates.size - 1, index_y2 + int(100 * (numpy.random.random() - 0.5))])
            print(nmax,index_x2, index_y2)
            zz = numpy.exp(numpy.sqrt((X - x_coordinates[index_x2]) ** 2 + (Y - y_coordinates[index_y2]) ** 2) / 2 / 30)
            if i == 0:
                Z = numpy.zeros((nmax,zz.shape[0],zz.shape[1]))
            Z[i,:,:] = zz

        #
        # initialize file
        #
        h5w = H5SimpleWriter.initialize_file("test2.h5", creator="h5_basic_writer.py")

        #
        # put data in file
        #

        # add some data at the main level
        h5w.add_key("stack shape", Z.shape)

        h5w.create_entry("mystackcollection", nx_default="mystack1")


        h5w.add_stack(numpy.arange(nmax), 1e3 * x_coordinates, 1e3 * y_coordinates, Z,
                      stack_name="stack1", entry_name="mystackcollection",
                      title_0="n", title_1="x", title_2="y")

        # the same data in a deepstack
        h5w.add_deepstack([numpy.arange(nmax), 1e3 * x_coordinates, 1e3 * y_coordinates], Z,
                          stack_name="stackDeep", entry_name="mystackcollection",
                          list_of_axes_titles=['n','x','y'])

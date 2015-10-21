import copy
import numpy


def set_scale_from_steps(np_array, initial_scale_value, scale_step):
    check_array(np_array)
    if scale_step <= 0.0: raise Exception("Scale Step must be > 0.0")

    return ScaledNPArray(np_array=np_array,
                         scale=numpy.arange(numpy.round(initial_scale_value, 12),
                                            numpy.round(initial_scale_value + (len(np_array)-1)*scale_step, 12),
                                            numpy.round(scale_step,12)))


def set_scale_from_range(np_array, min_scale_value, max_scale_value):
    check_array(np_array)
    if max_scale_value <= min_scale_value: raise Exception("Max Scale Value must be > Min Scale Vale")

    return set_scale_from_steps(np_array, min_scale_value, (max_scale_value - min_scale_value)/(len(np_array)-1))


def check_array(np_array):
    if np_array is None: raise Exception("Input Array is None")
    if not isinstance(np_array, (numpy.ndarray, numpy.generic)): raise Exception("Input Array is not a numpy array")
    if len(np_array.shape) != 1: raise Exception("Input Array is not a 1D numpy array")
    if len(np_array) == 0: raise Exception("Input Array is empty")


#######################################
#
# UTILITY CLASS
#
#######################################

class ScaledNPArray(object):

    np_array = None
    scale = None

    def __init__(self, np_array = numpy.zeros(0), scale = numpy.zeros(0)):
        if len(np_array) != len(scale): raise Exception("np_array and scale must have the same dimension")

        self.np_array=copy.deepcopy(np_array)
        self.scale=copy.deepcopy(scale)

        # Problem in comparison between float64 and numpy.float64:
        # reduce precision to avoid crazy research results
        for index in range(0, len(self.np_array)):
            self.np_array[index] = numpy.round(self.np_array[index], 12)
            self.scale[index] = numpy.round(self.scale[index], 12)

    def get_value(self, scale_value):
        scale_value = numpy.round(scale_value, 12)

        cursor_1 = numpy.where(self.scale >= scale_value)
        if len(cursor_1[0]) == 0: return None

        cursor_0 = numpy.where(self.scale < scale_value)
        if len(cursor_0[0]) == 0: return self.np_array[0]

        index_0 = cursor_0[0][len(cursor_0[0])-1]
        index_1 = cursor_1[0][0]

        x_0 = self.scale[index_0]
        x_1 = self.scale[index_1]
        y_0 = self.np_array[index_0]
        y_1 = self.np_array[index_1]

        return y_0 + ((y_1 - y_0) * (scale_value - x_0)/(x_1 - x_0))


if __name__ == "__main__":
     np_array = numpy.arange(15.0, 18.8, 0.2)
     scaled_array = set_scale_from_steps(np_array, 15.0, 0.2)

     print("interp", scaled_array.get_value(18.8))
     print("interp",  scaled_array.get_value(22.35))

     np_array = numpy.arange(15.0, 18.8, 0.2)
     scaled_array = set_scale_from_range(np_array, 15.0, 18.8)

     print("interp", scaled_array.get_value(18.8))
     print("interp",  scaled_array.get_value(22.35))

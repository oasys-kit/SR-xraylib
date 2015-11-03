import numpy

class DataExchangeObject(object):
    _content = None

    PROGRAM_NAME = "PROGRAM_NAME"
    WIDGET_NAME = "WIDGET_NAME"

    def __init__(self, program_name, widget_name):
        super().__init__()

        self._content = {DataExchangeObject.PROGRAM_NAME: program_name, DataExchangeObject.WIDGET_NAME: widget_name}

    def get_program_name(self):
        return self.get_content(DataExchangeObject.PROGRAM_NAME)

    def get_widget_name(self):
        return self.get_content(DataExchangeObject.WIDGET_NAME)

    def add_content(self, content_key="KEY", content_value=""):
        self._content[content_key] = content_value

    def get_content(self, content_key="KEY"):
        return self._content[content_key]

    def add_contents(self, content_keys=numpy.array(["KEY"]), content_values=numpy.array([""])):
        v_add_content = numpy.vectorize(self.add_content)
        v_add_content(content_keys, content_values)

    def get_contents(self, content_keys=numpy.array(["KEY"])):
        v_get_content = numpy.vectorize(self.get_content)
        return v_get_content(content_keys)

    def content_keys(self):
        return self._content.keys()[2:]
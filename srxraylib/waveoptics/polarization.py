from srxraylib import DeprecatedClassMeta

class Polarization(metaclass=DeprecatedClassMeta):
    _DeprecatedClassMeta__message="This class is deprected: use wofry and wofrylib instead"

    SIGMA = 0
    PI = 1
    TOTAL = 3
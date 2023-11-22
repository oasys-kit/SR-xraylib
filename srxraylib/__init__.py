__author__="luca rebuffi"

import functools
import warnings
import inspect

string_types = (type(b''), type(u''))

def deprecated(reason):
    if isinstance(reason, string_types):
        def decorator(function):

            if inspect.isfunction(function): message = "Call to deprecated function {name} ({reason})."
            else: raise ValueError("this decorator applies to functions only")

            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(message.format(name=function.__name__, reason=reason), category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)

                return function(*args, **kwargs)

            return wrapper

        return decorator
    elif inspect.isfunction(reason):
        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        function = reason

        if inspect.isfunction(function): message = "Call to deprecated function {name} ({reason})."
        else: raise ValueError("this decorator applies to functions only")

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(message.format(name=function.__name__), category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)

            return function(*args, **kwargs)

        return wrapper

    else:
        raise TypeError(repr(type(reason)))


class DeprecatedClassMeta(type):
    def __new__(cls, name, bases, classdict, *args, **kwargs):
        message = classdict.get('_DeprecatedClassMeta__message')
        alias   = classdict.get('_DeprecatedClassMeta__alias')

        if message is not None: warnings.warn("{} is deprecated, {}".format(name, message), DeprecationWarning, stacklevel=2)
        elif alias is not None:
            def new(cls, *args, **kwargs):
                alias = getattr(cls, '_DeprecatedClassMeta__alias')
                warnings.warn("{} has been renamed to {}, the alias will be removed in the future".format(cls.__name__,  alias.__name__), DeprecationWarning, stacklevel=2)

                return alias(*args, **kwargs)

            classdict['__new__'] = new
            classdict['_DeprecatedClassMeta__alias'] = alias

        fixed_bases = []

        for b in bases:
            message = getattr(b, '_DeprecatedClassMeta__message', None)
            alias   = getattr(b, '_DeprecatedClassMeta__alias', None)

            if message is not None: warnings.warn("{} is deprecated, {}".format(b.__name__, message), DeprecationWarning, stacklevel=2)
            elif alias is not None: warnings.warn("{} has been renamed to {}, the alias will be " "removed in the future".format(b.__name__, alias.__name__), DeprecationWarning, stacklevel=2)

            # Avoid duplicate base classes.
            b = alias or b
            if b not in fixed_bases: fixed_bases.append(b)

        fixed_bases = tuple(fixed_bases)

        return super().__new__(cls, name, fixed_bases, classdict, *args, **kwargs)

    def __instancecheck__(cls, instance):
        return any(cls.__subclasscheck__(c) for c in {type(instance), instance.__class__})

    def __subclasscheck__(cls, subclass):
        if subclass is cls: return True
        else:               return issubclass(subclass, getattr(cls, '_DeprecatedClassMeta__alias'))

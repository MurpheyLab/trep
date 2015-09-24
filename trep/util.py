import inspect
import _trep

def dynamics_indexing_decorator(type_string):
    ## This is a decorator for the dynamic functions in System and
    ## MidpointVI.  It converts the arguments in a call the
    ## appropriate index:
    ##
    ##  System.f_dddkdq(q, q1, q2)
    ##             -->  System.f_dddkq1(q.index, q1.k_index, q2.index)
    ##
    ## It adds support for passing None arguments by returning a slice
    ## instead:
    ## 
    ##  System.f_dddkdq(q, None, q2)
    ##             -->  System.f_dddkq1(q.index, slice(:), q2.index)
    ##
    ## so that a caller can pick out specific parts they want, or even
    ## take the whole array.
    ##
    ## We also do basic type checking.  This is un-Pythonic, but makes
    ## it much easier to find errors, like mixing up a kinematic and
    ## dynamic configuration variable.
    ##
    ## The type_string argument is a string with one character for
    ## each allowed argument:
    ##
    ##   q - any configuration variable
    ##   d - dynamic configuration variable
    ##   k - kinematic configuration variable
    ##   u - input variable
    ##   c - constraint variable
    ##

    for c in type_string:
        if c not in 'qdkuc':
            raise TypeError("invalid type spec: ", c)

    def decorator(func):
            
        def convert_args(self, *args):
            indices = []
            for type_, obj in zip(type_string, args):
                if obj is None:
                    indices.append(slice(None))
                else:
                    if type_ == 'q':
                        assert isinstance(obj, _trep._Config)
                        indices.append(obj.index)
                    elif type_ == 'd':
                        assert isinstance(obj, _trep._Config)
                        assert obj.kinematic == False
                        indices.append(obj.index)
                    elif type_ == 'k':
                        assert isinstance(obj, _trep._Config)
                        assert obj.kinematic == True
                        indices.append(obj.k_index)
                    elif type_ == 'u':
                        assert isinstance(obj, _trep._Input)
                        indices.append(obj.index)
                    elif type_ == 'c':
                        assert isinstance(obj, _trep._Constraint)
                        indices.append(obj.index)
                    else:
                        raise ValueError("internal error: invalid type")
            return func(self, *indices)

        # At this point, we could return convert_args() and be done,
        # but the resulting functions would all be be called
        # "convert_args", have the same anonymous structure, and no
        # doc string.  Instead, we want the returned function to
        # appear identical to the original.  We use inspect to find
        # out the arguments of func and then create a new function
        # with the same name and arguments that calls convert_args.
        # Basically a wrapper for our wrapper.

        spec = inspect.getargspec(func)
        if spec.varargs != None:
            raise TypeError("variable args are not supported")
        if spec.keywords != None:
            raise TypeError("keywords are not supported")
        if spec.defaults is None or len(spec.args) != len(spec.defaults)+1:
            raise TypeError("missing default None arguments")
        if False in [d is None for d in spec.defaults]:
            raise TypeError("default arguments should be None")
        if len(type_string) +1 != len(spec.args):
            raise TypeError("type_string is incorrect length")
        signature = [spec.args[0]] + ['%s=None' % a for a in spec.args[1:]]
        signature = ", ".join(signature)

        src =  "def %s(%s):\n" % (func.__name__, signature)
        src += "    return convert_args(%s)" % ','.join(spec.args)

        context = {'func' : func, 'convert_args' : convert_args}
        exec src in context

        wrapper = context[func.__name__]
        wrapper.__dict__ = func.__dict__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

def get_include():
    """
    Return the directory that contains the trep \\*.h header files.
    Extension modules that need to compile against trep should use this
    function to locate the appropriate include directory.
    """
    import trep, os.path
    return os.path.join(os.path.dirname(trep.__file__), '_trep')


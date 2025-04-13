from mh.core import StaticClass, AutoStatic, build_module_getter_callback

class mu_x(metaclass=StaticClass):
    class __call__(AutoStatic):
        def dummy(val):
            return val

class peak_time(metaclass=StaticClass):
    class __call__(AutoStatic):
        def inverse_freq(*, factor, freq):
            return factor / freq
        
get = build_module_getter_callback(globals())
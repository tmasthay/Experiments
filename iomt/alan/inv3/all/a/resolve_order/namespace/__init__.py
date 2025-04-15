from mh.core import StaticClass, AutoStatic, build_module_getter_callback

class check(AutoStatic):
    def always_true(c):
        return True
    
    def always_false(c):
        return False

get =  build_module_getter_callback(globals())
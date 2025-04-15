import torch
import torch.optim as optim
from mh.core import StaticClass, AutoStatic, build_module_getter_callback

class _helpers(metaclass=StaticClass):
    def get_opt_params(params):
        return [p for p in params.values() if hasattr(p, 'requires_grad') and p.requires_grad]
    
    def get_opt_params_and_validate(params):
        res = _helpers.get_opt_params(params)
        if not res:
            raise ValueError("None of the parameters require gradients; nothing to optimize.")
        return res
    
    def str_to_class(class_name):
        d = {
            'sgd': optim.SGD,
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'adamax': optim.Adamax,
            'adagrad': optim.Adagrad,
            'adamax': optim.Adamax,
            'adamw': optim.AdamW,
            'rmsprop': optim.RMSprop,
            'rprop': optim.Rprop,
            'asgd': optim.ASGD,
            'lbfgs': optim.LBFGS
        }

    
class __call__(AutoStatic):
    def simple(*, cls, params, lr, **kwargs):
        cls = _helpers.str_to_class(cls.lower()) # case insensitive
        params = _helpers.get_opt_params_and_validate(params)
        return cls(params, lr=lr, **kwargs)
    

get = build_module_getter_callback(globals())
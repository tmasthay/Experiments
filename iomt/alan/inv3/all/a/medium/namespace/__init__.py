import torch
from mh.core import StaticClass, AutoStatic, build_module_getter_callback

class _helper(metaclass=StaticClass):
    def simple(model, shape, trainable, device):
        if isinstance(model, str):
            v = torch.load(model, map_location=device)
        elif isinstance(model, (int, float)):
            v = torch.full(shape, model, device=device, dtype=torch.float32)
        elif isinstance(model, list):
            v = torch.tensor(model, device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported velocity model format: {type(model)}")
        
        v.requires_grad_(trainable)  # we are not optimizing velocity in this task
        return v
    
class vp(metaclass=StaticClass):
    class __call__(AutoStatic):
        def simple(*, model, shape, trainable, device):
            return _helper.simple(model, shape, trainable, device)

class vs(vp):
    pass # all the same for now

class rho(vp):
    pass # all the same for now
    
get = build_module_getter_callback(globals())
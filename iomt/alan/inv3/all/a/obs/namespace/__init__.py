import torch
from mh.core import AutoStatic, StaticClass, build_module_getter_callback
from os.path import join as pj, exists

class __call__(AutoStatic):
    def simple(*, forward, vp, device, path):
        true_path = pj(__file__, '..', path.replace('.pt', '') + '.pt')
        if( exists(true_path) ):
            return torch.load(true_path).to(device=device)
        
        u = forward(vp)
        torch.save(u.detach().to('cpu'), true_path)
        return u

get = build_module_getter_callback(globals())
        
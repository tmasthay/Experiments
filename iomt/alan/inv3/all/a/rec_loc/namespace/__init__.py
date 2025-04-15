from mh.core import AutoStatic, StaticClass, build_module_getter_callback
from itertools import product
from time import time
import torch 


class __call__(AutoStatic):
    def cartesian(*, y, x, the_type=torch.int, device):
        return torch.tensor(list(product(y, x)), dtype=the_type, device=device)
    
    def uniform(*, depth, left, right, device, num_recs, ny, nx):
        assert num_recs < nx, f"{num_recs=} must be less than {nx=}"
        assert 0 <= num_recs < nx, f"{num_recs=} must be in [0, {nx=})"
        assert 0 <= depth < ny, f"{depth=} must be in [0, {ny=})"
        
        y = torch.tensor([depth], dtype=torch.int, device=device)   
        
        x = torch.linspace(
            left, right, num_recs, dtype=torch.int, device=device
        )
        return torch.cartesian_prod(y, x).to(device=device)
        
    def uniform_scaled(*, depth, left, right, num_recs, ny, nx, device):
        depth_int = int(depth * ny)
        left_int = int(left * nx)
        right_int = int(right * nx)
        num_recs_int = int(num_recs * nx)
        return __call__.uniform(
            depth=depth_int,
            left=left_int,
            right=right_int,
            num_recs=num_recs_int,
            ny=ny,
            nx=int(nx * (right - left)),
            device=device
        )
        

get = build_module_getter_callback(globals())
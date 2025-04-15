from mh.core import AutoStatic, StaticClass, build_module_getter_callback
from itertools import product
import torch 


class __call__(AutoStatic):
    
    def uniform(*, ly, ry, nry, ny, lx, rx, nrx, nx, device, nshots=1):
        assert nry < ny, f"{nry=} must be less than {ny=}"
        assert 0 <= nry < ny, f"{nry=} must be in [0, {ny=})"
        assert nrx < nx, f"{nrx=} must be less than {nx=}"
        assert 0 <= nrx < nx, f"{nrx=} must be in [0, {nx=})"
        
        
        y = torch.linspace(
            ly, ry, nry, dtype=torch.int, device=device)  
        
        x = torch.linspace(
            lx, rx, nrx, dtype=torch.int, device=device
        )
        res = torch.cartesian_prod(y, x).to(device=device).expand(
            nshots, -1, -1
        )
        return res
        
    def uniform_scaled(*, ly, ry, nry, ny, lx, rx, nrx, nx, device, nshots=1):
        nry_int = int(nry * ny)
        nrx_int = int(nrx * nx)
        return __call__.uniform(
            ly=ly,
            ry=ry,
            nry=nry_int,
            ny=int(ny * (ry - ly)),
            lx=lx,
            rx=rx,
            nrx=nrx_int,
            nx=int(nx * (rx - lx)),
            nshots=nshots,
            device=device
        )
        
        

get = build_module_getter_callback(globals())
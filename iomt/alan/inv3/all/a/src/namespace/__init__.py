from mh.core import StaticClass, AutoStatic, build_module_getter_callback
import torch
import deepwave as dw

class FullGridContinuousSource(torch.nn.Module):
    def __init__(self, *, src_locs, disc, continuation, loc_parameterization, src_amp, device):
        super().__init__()
        self.continuation = continuation(src_locs=src_locs, device=device, disc=disc)
        self.loc_parameterization = torch.nn.Parameter(
            loc_parameterization, requires_grad=True).to(device=device)
        self.src_amp = torch.nn.Parameter(src_amp, requires_grad=True).to(device=device)
        self.register_parameter('loc_parameterization', self.loc_parameterization)

    def forward( self ):
        mask = self.continuation(self.loc_parameterization)
        assert mask.shape == self.src_amp.shape, f"{mask.shape=} {self.src_amp.shape=}"
        return mask * self.src_amp
    
class _helpers(metaclass=StaticClass):
    def linear(*, src_locs, disc, device):
        nshots, _, ndims = src_locs.shape
        
        assert src_locs.dtype == torch.int, f"{src_locs.dtype=}, expected int"
        assert src_locs.ndim == 3, f"{src_locs.ndim=}, expected 3"
        assert src_locs.shape[0] == nshots, f"{src_locs.shape=}, expected {nshots=}"
        assert src_locs.shape[2] == 2, f"{src_locs.shape=}, expected {ndims=}"
        
        ny, nx = disc.ny, disc.nx
        dy, dx = disc.dy, disc.dx
        min_y, min_x = 0, 0
        max_y, max_x = dy * (ny - 1), dx * (nx - 1)
        

        
        locs_float = src_locs.float()
        locs_float[..., 0] = locs_float[..., 0] * dy
        locs_float[..., 1] = locs_float[..., 1] * dx
        max_dist = torch.sqrt(
            torch.tensor((max_y - min_y) ** 2 + (max_x - min_x) ** 2)
        )

        def linear_continuation(loc_parameterization):
            cy, cx = loc_parameterization[0], loc_parameterization[1]
            cy_abs = min_y + cy * (max_y - min_y)
            cx_abs = min_x + cx * (max_x - min_x)
            y = locs_float[..., 0]
            x = locs_float[..., 1]
            dist = torch.sqrt((y - cy_abs) ** 2 + (x - cx_abs) ** 2)
            mask = 1.0 - (dist / (max_dist + 1e-6))
            return mask.clamp(min=0)
    
        return linear_continuation

class amp(metaclass=StaticClass):
    class rt(AutoStatic): 
        class __call__(AutoStatic):
            def linear(*, src_locs, disc, device):
                return FullGridContinuousSource(
                    src_locs=src_locs,
                    disc=disc,
                    continuation=_helpers.linear,
                    device=device
                )
    
    class truth(AutoStatic):
        class __call__(AutoStatic):
            def ricker_dirac(*, disc, loc, peak_time_factor, freq, src_locs, device):
                ny, nx, nt = disc.ny, disc.nx, disc.nt
                dy, dx, dt = disc.dy, disc.dx, disc.dt
                
                peak_time = peak_time_factor * freq
                
                u = dw.wavelets.ricker(
                    freq=freq,
                    length=nt,
                    dt=dt,
                    peak_time=peak_time,
                ).to(device=device)
                
                src_amp = torch.zeros((src_locs.shape[0], src_locs.shape[1], nt), device=device)
                loc_idx = int(loc * src_amp.shape[1])
                src_amp[:, loc_idx, :] = u
                return src_amp
                


get = build_module_getter_callback(globals())

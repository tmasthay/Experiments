import deepwave as dw
from mh.core import StaticClass, AutoStatic, build_module_getter_callback
from misfit_toys.types import UnaryPartial as UF, KwEnforcer
import torch

class ParameterizatedAmpsAcoustic(torch.nn.Module):
    def __init__(self, *, src_locs, disc, continuation, device):
        super().__init__()
        self.continuation = continuation(src_locs=src_locs, device=device, disc=disc)

class __call__(AutoStatic):
    def acoustic(*, disc, src_amp, src_loc, rec_loc, **kwargs):
        return UF(
            callback=dw.scalar,
            grid_spacing=[disc.dy, disc.dx],
            dt=disc.dt,
            nt=disc.nt,
            src_amp=src_amp,
            src_loc=src_loc,
            rec_loc=rec_loc,
            **kwargs
        )
        
    def elastic(*, disc, **kwargs):
        return KwEnforcer(
            callback=dw.elastic,
            grid_spacing=[disc.dy, disc.dx],
            dt=disc.dt,
            nt=disc.nt,
            required_keys=['lamb', 'mu', 'rho'],
            **kwargs
        )
        


get = build_module_getter_callback(globals())

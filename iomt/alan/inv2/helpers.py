import torch
from misfit_toys.utils import tslice


class EasyW1Loss(torch.nn.Module):
    def __init__(self, ref_data, *, renorm=None, dim=-1, eps=1e-8):
        super().__init__()
        if renorm is None:
            renorm = torch.abs

        self.renorm = renorm
        self.dim = dim
        self.eps = eps
        self.cdf = self.__cdf__(ref_data, renorm=renorm, eps=eps, dim=dim)
        # input(f'{self.cdf.shape=}')

    @staticmethod
    def __cdf__(data, *, renorm, dim=-1, eps=1e-8):
        pdf = renorm(data)
        assert pdf.min() >= 0.0, f'{pdf.min()=}, should be >= 0.0'

        v = torch.cumulative_trapezoid(pdf, dim=dim)
        assert torch.isnan(v).sum() == 0, f'{v=}'
        divider = tslice(v, dims=[dim]).unsqueeze(dim)
        assert divider.min() > 0.0, f'{divider.min()=}, should be > 0.0'
        u = v / (eps + tslice(v, dims=[dim]).unsqueeze(dim=dim))
        assert u.max() <= 1.0, f'{u.max().item()=}, should be <= 1.0'
        assert u.min() >= 0.0, f'{u.min().item()=}, should be >= 0.0'
        return u

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        lcl_cdf = EasyW1Loss.__cdf__(
            data, renorm=self.renorm, dim=self.dim, eps=self.eps
        )
        # raise RuntimeError(f'{self.cdf.shape=}, {lcl_cdf.shape=}')
        # diff = (lcl_cdf - self.cdf).abs()
        # return torch.sum(diff**2, dim=self.dim)
        diff = lcl_cdf - self.cdf
        integrand = diff**2
        res = torch.mean(integrand, dim=self.dim)

        # consider refactoring later if you want something
        # more general, but for now this is fine
        res_flat = res.view(res.shape[0], -1)
        return res_flat.mean(dim=1)
        # raise RuntimeError(f'{torch.mean(integrand, dim=self.dim).shape=}')

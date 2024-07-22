import hydra
from omegaconf import OmegaConf, DictConfig
from mh.core import (
    exec_imports,
    DotDict,
    cfg_import,
    torch_stats,
    set_print_options,
)
from typing import Callable
from misfit_toys.utils import apply_all
from submods.config.custom_types import ConfigurableUnaryFunction as CUF

set_print_options(callback=torch_stats('all'))


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig):
    self_read = apply_all(
        exec_imports(
            DotDict(OmegaConf.to_container(cfg.self_read, resolve=True))
        ).self_ref_resolve()
    )
    d = self_read(cfg)
    print(f'{type(d)=}\n\n{d=}\n\n')


if __name__ == "__main__":
    main()

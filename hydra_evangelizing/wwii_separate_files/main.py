import hydra
from omegaconf import DictConfig
from helpers import hydra_kw
from rich import print as rprint

config_path = 'config'
config_name = 'entry'


@hydra_kw(use_cfg=True)
def print_case(cfg):
    rprint(cfg)


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg: DictConfig) -> None:
    nested_index = cfg.theater.split('.')
    curr = cfg.case_map
    for idx in nested_index:
        curr = curr[idx]

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    print_case(config_path=config_path, config_name=curr)


if __name__ == "__main__":
    main()

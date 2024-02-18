import hydra
from omegaconf import DictConfig

config_path = 'config'
config_name = 'entry'


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(cfg: DictConfig) -> None:
    nested_index = cfg.theater.split('.')
    curr = cfg[nested_index[0]]
    nested_index = nested_index[1:]
    for idx in nested_index:
        curr = curr[idx]

    print(f'Theater: {cfg.theater}')
    print(curr)


if __name__ == "__main__":
    main()

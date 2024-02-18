import hydra
from omegaconf import DictConfig


def out_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)
    print(f'Current out_dir: {out_dir()}')


if __name__ == "__main__":
    main()

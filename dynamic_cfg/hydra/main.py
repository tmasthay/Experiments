import hydra
from omegaconf import OmegaConf, DictConfig

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    main()
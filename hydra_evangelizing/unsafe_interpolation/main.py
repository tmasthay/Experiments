import hydra
from omegaconf import DictConfig
from helpers import DotDict, convert_dictconfig, format_with_black
import numpy as np
import matplotlib.pyplot as plt
import os


np.set_printoptions(threshold=6)


def out_dir(*args):
    return os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, *args
    )


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg, self_ref_resolve=False)
    c.t = np.linspace(*c.t)
    return c.self_ref_resolve(10, globals(), locals())


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(cfg: DictConfig) -> None:
    c = preprocess_cfg(cfg)

    if c.get("verbose", False):
        print(format_with_black(c.str()))

    exc = ['style', 'save']
    for k, v in c.plt.filter(exclude=exc).items():
        plt.plot(c.t, v.data, label=v.label, **v.style)
    plt.legend()
    plt.title('You could further configure this title if you really wanted')
    plt.savefig(out_dir(c.plt.save.filename))

    print(
        f"Saved plot to {out_dir(c.plt.save.filename)} <--- CTRL+click to open"
        " in VS code"
    )


if __name__ == "__main__":
    main()

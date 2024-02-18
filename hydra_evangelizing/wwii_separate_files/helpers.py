from hydra import compose, initialize
from functools import wraps
import os


def hydra_kw(*, use_cfg=False, protect_kw=True, transform_cfg=None):
    if not use_cfg and transform_cfg:
        UserWarning(
            'use_cfg is False with non-null transform_cfg -> transform_cfg will'
            ' be ignored'
        )

    def decorator(f):
        @wraps(f)
        def wrapper(
            *args,
            config_path=None,
            config_name=None,
            version_base=None,
            overrides=None,
            return_hydra_config=False,
            **kw,
        ):
            if config_path is None or config_name is None:
                cfg = {}
            else:
                config_path = os.path.relpath(
                    config_path, os.path.dirname(__file__)
                )
                with initialize(
                    config_path=config_path, version_base=version_base
                ) as cfg:
                    cfg = compose(
                        config_name=config_name,
                        overrides=overrides,
                        return_hydra_config=return_hydra_config,
                    )

            overlapping_keys = set(cfg.keys()).intersection(set(kw.keys()))
            for key in overlapping_keys:
                kw[key] = cfg[key]
                if protect_kw:
                    del cfg[key]
            if use_cfg:
                if transform_cfg is not None:
                    cfg = transform_cfg(cfg)
                return f(cfg, *args, **kw)
            else:
                return f(*args, **kw)

        return wrapper

    return decorator

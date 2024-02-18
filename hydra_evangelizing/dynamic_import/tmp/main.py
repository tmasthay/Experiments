import importlib.util
import sys
import os
import hydra
from omegaconf import DictConfig
from helpers import DotDict, convert_dictconfig, format_with_black


class Paths:
    path = 'cfg'
    cfg = 'cfg'


def dyn_import(*, path, mod, func=None):
    if not path.startswith('/'):
        path = os.path.join(os.getcwd(), path)
    if not path.endswith('.py'):
        path = os.path.join(path, f'{mod}.py')

    spec = importlib.util.spec_from_file_location(mod, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod] = module
    spec.loader.exec_module(module)
    obj = module
    if func is not None:
        obj = getattr(module, func)
    return obj


def cfg_import(s, *, root=None, delim='|'):
    info = s.split(delim)
    root = root or os.getcwd()
    if len(info) == 1:
        path, mod, func = root, info[0], None
    elif len(info) == 2:
        path, mod, func = root, info[0], info[1]
    else:
        path, mod, func = info

    if func is not None and func.lower() in ['none', 'null', '']:
        func = None

    path = os.path.abspath(path)
    return dyn_import(path=path, mod=mod, func=func)


def exec_imports(d: DotDict, *, root=None, delim='|', import_key='dimport'):
    q = [('', d)]
    root = root or Paths.path
    while q:
        prefix, curr = q.pop(0)
        for k, v in curr.items():
            if isinstance(v, DotDict) or isinstance(v, dict):
                q.append((f'{prefix}.{k}' if prefix else k, v))
            elif isinstance(v, list) and len(v) > 0 and v[0] == import_key:
                lcl_root = os.path.join(root, *prefix.split('.'))
                d[f'{prefix}.{k}'] = cfg_import(
                    v[1], root=lcl_root, delim=delim
                )
    return d


def preprocess_cfg(cfg: DictConfig) -> DotDict:
    c = convert_dictconfig(cfg)
    c = exec_imports(c)
    return c


@hydra.main(config_path=Paths.path, config_name=Paths.cfg, version_base=None)
def main(cfg: DictConfig):
    c = preprocess_cfg(cfg)
    for k, v in c.items():
        print(v.prefix, end=v.get('end', '\n'))
        v.callback(*v.args, **v.kwargs)
        print(v.get('suffix', ''), end=v.get('end', '\n'))


if __name__ == '__main__':
    main()

import copy
from omegaconf import OmegaConf
import black


def format_with_black(
    code: str,
    line_length: int = 80,
    preview: bool = True,
    magic_trailing_comma: bool = False,
    string_normalization: bool = False,
) -> str:
    try:
        mode = black.FileMode(
            line_length=line_length,
            preview=preview,
            magic_trailing_comma=magic_trailing_comma,
            string_normalization=string_normalization,
        )
        formatted_code = black.format_str(code, mode=mode)
        return formatted_code
    except black.NothingChanged:
        return code


class DotDict:
    def __init__(self, d, self_ref_resolve=False):
        D = copy.deepcopy(d)
        if type(d) is DotDict:
            self.__dict__.update(d.__dict__)
        else:
            for k, v in D.items():
                if type(v) is dict:
                    D[k] = DotDict(v, self_ref_resolve=False)
                elif type(v) is list:
                    D[k] = [
                        (
                            DotDict(e, self_ref_resolve=False)
                            if type(e) is dict
                            else e
                        )
                        for e in v
                    ]
            self.__dict__.update(D)
        if self_ref_resolve:
            self.self_ref_resolve()

    def set(self, k, v):
        self.deep_set(k, v)

    def get(self, k, default_val=None):
        try:
            return self.deep_get(k)
        except KeyError:
            return default_val

    def __setitem__(self, k, v):
        self.deep_set(k, v)

    def __getitem__(self, k):
        return self.deep_get(k)

    def __setattr__(self, k, v):
        if isinstance(v, dict):
            v = DotDict(v)
        self.__dict__[k] = v

    def getd(self, k, v):
        return self.__dict__.get(k, v)

    def setdefault(self, k, v):
        self.__dict__.setdefault(k, v)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def update(self, d):
        self.__dict__.update(DotDict.get_dict(d))

    def str(self):
        return str(self.__dict__)

    def dict(self):
        return self.__dict__

    def __str__(self):
        return self.str()

    def __repr__(self):
        return self.str()

    def deep_get(self, k):
        d = self.__dict__
        keys = k.split('.')
        for key in keys:
            d = d[key]
        return d

    def deep_set(self, k, v):
        d = self.__dict__
        keys = k.split('.')
        for key in keys[:-1]:
            try:
                d = d[key]
            except KeyError:
                d[key] = DotDict({})
                d = d[key]
        d[keys[-1]] = v

    def has_self_ref(self):
        d = self.__dict__
        q = [d]
        while q:
            d = q.pop()
            for k, v in d.items():
                if isinstance(v, DotDict):
                    q.append(v)
                elif isinstance(v, dict):
                    q.append(v)
                elif isinstance(v, str):
                    if 'self' in v or 'eval(' in v:
                        return True
        return False

    def self_ref_resolve(self, max_passes=10, glb=None, lcl=None):
        lcl.update(locals())
        glb.update(globals())
        passes = 0
        while passes < max_passes and self.has_self_ref():
            d = self.__dict__
            q = [d]
            while q:
                d = q.pop()
                for k, v in d.items():
                    if isinstance(v, DotDict):
                        q.append(v)
                    elif isinstance(v, dict):
                        d[k] = DotDict(v)
                        q.append(d[k])
                    elif isinstance(v, str):
                        if 'self' in v:
                            d[k] = eval(v, glb, lcl)
                        elif 'eval(' in v:
                            d[k] = eval(v[5:-1], glb, lcl)
            passes += 1
        if passes == max_passes:
            raise ValueError(
                f"Max passes ({max_passes}) reached. self_ref_resolve failed"
            )
        return self

    def filter(self, exclude=None, include=None, relax=False):
        keys = set(self.keys())
        exclude = set() if exclude is None else set(exclude)
        include = keys if include is None else set(include)
        if not relax:
            if not include.issubset(keys):
                raise ValueError(
                    f"include={include} contains keys not in d={keys}"
                )
            if not exclude.issubset(keys):
                raise ValueError(
                    f"exclude={exclude} contains keys not in d={keys}...use"
                    " relax=True to ignore this error"
                )
            return DotDict({k: self[k] for k in include.difference(exclude)})
        else:
            include = include.intersection(keys)
            exclude = exclude.intersection(include)
            return DotDict(
                {k: self.get(k, None) for k in include.difference(exclude)}
            )


def convert_dictconfig(obj, self_ref_resolve=False):
    return DotDict(
        OmegaConf.to_container(obj, resolve=True),
        self_ref_resolve=self_ref_resolve,
    )

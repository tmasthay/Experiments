import importlib
import sys
import bpy
import math
import numpy as np
from typing import Any
import copy
from functools import wraps
import os
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
import hydra


class LocalNamespace:
    types = {
        "int": int,
        "float": float,
        "complex": complex,
        "float32": np.float32,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "str": str,
        "bool": bool,
    }


# def easy_main(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         hydra_main = hydra.main(
#             config_path='.', config_name='config.yaml', version_base=None
#         )
#         return hydra_main(func)(*args, **kwargs)

#     return wrapper


def easy_main(
    preprocess_func=None,
    *,
    config_path=None,
    config_name='config.yaml',
    version_base=None,
):
    config_path = config_path or os.getcwd()
    if preprocess_func is None:
        preprocess_func = (
            lambda cfg: cfg
        )  # Default to identity function if no preprocess function is provided

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply hydra_main_decorator to a new inner function
            @hydra.main(
                config_path=config_path,
                config_name=config_name,
                version_base=version_base,
            )
            def hydra_main(cfg: DictConfig, *args_dummy, **kwargs_dummy):
                # Preprocess cfg using the preprocess function
                cfg = preprocess_func(cfg)
                return func(cfg, *args_dummy, **kwargs_dummy)

            # Call the hydra_main function with args and kwargs
            return hydra_main(*args, **kwargs)

        return wrapper

    return decorator


class DotDict:
    def __init__(self, d):
        if type(d) is DotDict:
            self.__dict__.update(d.__dict__)
        else:
            self.__dict__.update(d)

    def set(self, k, v):
        self.__dict__[k] = v

    def get(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        self.set(k, v)

    def __getitem__(self, k):
        return self.get(k)

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

    def has(self, k):
        return hasattr(self, k)

    def has_all(self, *keys):
        return all([self.has(k) for k in keys])

    def has_all_type(self, *keys, lcl_type=None):
        return all(
            [self.has(k) and type(self.get(k)) is lcl_type for k in keys]
        )

    def update(self, d):
        self.__dict__.update(DotDict.get_dict(d))

    def dict(self):
        return self.__dict__

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def get_dict(d):
        if isinstance(d, DotDict):
            return d.dict()
        else:
            return d


def convert_config(obj, list_protect="list_protect", dtype=np.float32):
    if isinstance(obj, DictConfig):
        obj = DotDict(obj.__dict__['_content'])
    elif isinstance(obj, dict):
        obj = DotDict(obj)

    if isinstance(obj, DotDict):
        if list(obj.keys()) == ['type', 'value']:
            return LocalNamespace.types[obj['type']](obj['value']._value())

        if 'default_type' not in obj.keys():
            obj['default_type'] = dtype
        else:
            obj['default_type'] = LocalNamespace.types[obj['default_type']]

        for key, value in obj.items():
            if isinstance(value, AnyNode):
                obj[key] = obj['default_type'](value._value())

        for key, value in obj.items():
            if key != list_protect:
                obj[key] = convert_config(value, list_protect)
    elif isinstance(obj, list) or isinstance(obj, ListConfig):
        if type(obj[0]) == str:
            return np.array(obj[1:], dtype=LocalNamespace.types[obj[0]])
        else:
            return np.array(obj, dtype=dtype)
    return obj


def numpy_lists(name=0, list_protect="list_protect"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Retrieve the configuration object
            if isinstance(name, int):
                if name >= len(args) or not isinstance(args[name], DictConfig):
                    raise ValueError(
                        "Configuration object not found at the specified index"
                        " in args."
                    )
                # Modify args in place
                args = list(args)  # Convert args to a mutable list
                args[name] = convert_config(args[name], list_protect)
                args = tuple(args)  # Convert back to tuple
            elif isinstance(name, str):
                if name not in kwargs or not isinstance(
                    kwargs[name], DictConfig
                ):
                    raise ValueError(
                        "Configuration object not found for the specified key"
                        " in kwargs."
                    )
                # Modify kwargs in place
                kwargs[name] = convert_config(kwargs[name], list_protect)
            else:
                raise ValueError(
                    "Argument 'name' must be either an integer or a string."
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def none_handler(defaults):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for key, handler in defaults.items():
                if kwargs.get(key) is None:
                    kwargs[key] = handler()
            return func(*args, **kwargs)

        return wrapper

    return decorator


context_handler = none_handler({'C': lambda: bpy.context})
data_handler = none_handler({'D': lambda: bpy.data})
bpy_handler = none_handler({'C': lambda: bpy.context, 'D': lambda: bpy.data})


def expand_shape(x, l):
    s = len(x)
    if s > l:
        return x
    else:
        return (x * int(np.ceil(l / s)))[:l]


def cycle(*, seq: list, inter_seq: list):
    l = math.lcm(len(seq), len(inter_seq))
    seq, inter_seq = expand_shape(seq, l), expand_shape(inter_seq, l)

    def elem(i):
        if i == l - 1:
            return [seq[i], inter_seq[i], seq[0]]
        return [seq[i], inter_seq[i], seq[i + 1]]

    return [elem(i) for i in range(l)]


def delete_existing_keyframes(obj):
    if obj.animation_data:
        obj.animation_data_clear()


def beat_sequence(*, seq: np.ndarray, beat_unit: int, num_units: int):
    increments = np.arange(1, num_units) * beat_unit
    res = np.vstack([seq + increment for increment in increments])
    return np.concatenate((seq, res.flatten()))


def beats_to_frames(beats, bpm, fps):
    exact_frames = beats * fps * 60 / bpm
    return np.round(exact_frames).astype(int)


def expand_frames(frames, width=1):
    if width == 0:
        return frames
    res = []
    for frame in frames:
        res.append([frame + i for i in range(-width, width + 1)])
    return res


def collapse(arr):
    res = []
    for e in arr:
        if isinstance(e, list):
            res.extend(e)
        else:
            res.append(e)
    return res


def beat_subdiv(*, seq, subdivs, start_beat):
    unit = 1.0 / subdivs
    res = copy.deepcopy(seq)
    for i in range(len(res)):
        res[i] = unit * (res[i] + start_beat + i * subdivs)
    return res.flatten()


def sim_join(*, frames: list, vals: list):
    frames, vals = list(frames), list(vals)
    if len(frames) > len(vals):
        vals = (len(frames) // len(vals) + 1) * vals
        vals = vals[: len(frames)]
    else:
        frames = (len(vals) // len(frames) + 1) * frames
        frames = frames[: len(vals)]
    return list(zip(frames, vals))


def blip(*, seq: list, idx: int, inplace=True):
    if inplace:
        if idx <= 0:
            return seq

        val = seq[idx - 1][1]
        seq.insert(idx, (seq[idx][0] - 1, val))
        idx += 1
        seq.insert(idx + 1, (seq[idx][0] + 1, val))
        return seq
    else:
        if idx <= 0:
            return None
        frame = seq[idx][0]
        val = seq[idx - 1][1]
        return [(frame - 1, val), (frame, seq[idx][1]), (frame + 1, val)]


def blip_init(*, seq: list, init_val: Any):
    seq.insert(0, (seq[0][0] - 1, init_val))
    seq.insert(0, (0, init_val))
    return seq


def blip_all(*, seq: list, init_val: Any):
    tokens = [blip(seq=seq, idx=i, inplace=False) for i in range(1, len(seq))]
    res = []
    [res.extend(token) for token in tokens]
    res = blip_init(seq=res, init_val=init_val)
    return res


def get_uniform_frames(*, start, interval, duration, fps):
    frames = []
    current_time = start
    accumulated_error = 0.0

    while current_time <= start + duration:
        exact_frame = current_time * fps
        accumulated_error += exact_frame - math.floor(exact_frame)

        if accumulated_error >= 1.0:
            frame = math.ceil(exact_frame)
            accumulated_error -= 1.0
        else:
            frame = math.floor(exact_frame)

        frames.append(frame)
        current_time += interval

    return frames


def get_uniform_transformations(*, start, interval, duration, fps, seq):
    frames = get_uniform_frames(
        start=start, interval=interval, duration=duration, fps=fps
    )
    seq_repeats = np.repeat(seq, len(frames) // len(seq) + 1)[: len(frames)]
    return list(zip(frames, seq_repeats))


@context_handler
def animate_property(obj, prop, stamps, C=None):
    if not hasattr(obj, prop):
        raise ValueError(f"The object does not have a property '{prop}'.")

    for frame_number, value in stamps:
        C.scene.frame_set(frame_number)

        if isinstance(value, (int, float)):
            value = (value, value, value)
        setattr(obj, prop, value)

        obj.keyframe_insert(data_path=prop)


def animate(d: dict, *, obj='active'):
    if obj == 'active':
        obj = bpy.context.active_object
    for prop, stamps in d.items():
        animate_property(obj, prop, stamps)


# def no_pip_import(file_path, rel_root=True):
#     if not file_path.startswith(os.sep) and rel_root:
#         root = os.path.dirname(__file__)
#         file_path = os.path.join(root, file_path)
#     # Extract the directory and module name from the file path
#     directory, module_name = os.path.split(file_path)
#     module_name = os.path.splitext(module_name)[0]

#     # Add the directory to sys.path
#     sys.path.append(directory)

#     # Import the module
#     module = importlib.import_module(module_name)

#     # Call the 'main' function from the module
#     if hasattr(module, 'main') and callable(module.main):
#         return module.main
#     else:
#         print(f"No callable 'main' function found in {file_path}")


def demo_subdiv():
    seq = np.array(
        [[1.0, 3], [2, 3], [1, 2], [1, 3], [2, 3], [1, 2], [1, 3], [2, 3]]
    )
    print(f'original:\n{seq}')
    final = beat_subdiv(seq=seq, subdivs=4, start_beat=0)
    print(f'beat_subdiv:\n{final}')
    return final, seq


def demo_beat_frames(x):
    beats = beat_sequence(seq=x, beat_unit=int(np.ceil(max(x))), num_units=4)
    frames = beats_to_frames(beats=beats, bpm=120, fps=32)

    print(f'beat_sequence:\n{beats}')
    print(f'beats_to_frames:\n{frames}')
    return frames


def get_frames(cfg: DictConfig):
    cfg = convert_config(cfg)
    cfg.subdiv_seq = beat_subdiv(
        seq=cfg.seq, subdivs=cfg.subdivs, start_beat=cfg.start_beat
    )
    cfg.beats = beat_sequence(
        seq=cfg.subdiv_seq, beat_unit=cfg.beat_unit, num_units=cfg.num_units
    )
    cfg.frames = beats_to_frames(beats=cfg.beats, bpm=cfg.bpm, fps=cfg.fps)
    return cfg


if __name__ == "__main__":
    x, original = demo_subdiv()
    y = demo_beat_frames(x)
    print(
        '(beat_to_frames % 16) /'
        f' 4:\n{np.array((y % 16) / 4, dtype=int).reshape(-1,2)}'
    )

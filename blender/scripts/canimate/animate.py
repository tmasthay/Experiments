import bpy
import math
import numpy as np
from typing import Any
from collections import OrderedDict
import copy
from functools import wraps


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


def beat_sequence(*, seq: list, beat_unit: int, num_units: int):
    res = copy.deepcopy(seq)
    shifted_vals = copy.deepcopy(seq)
    for _ in range(num_units - 1):
        shifted_vals = [shifted_vals[i] + beat_unit for i in range(len(seq))]
        res += shifted_vals
    return res


def beats_to_frames(beats, bpm, fps):
    fpm = fps * 60  # Frames per minute
    fpb = fpm / bpm  # Frames per beat
    accumulated_error = 0.0
    beat_to_frame_mapping = []

    for beat in beats:
        # Calculate the exact frame for the current beat
        exact_frame = beat * fpb
        # accumulated_error += exact_frame - math.floor(exact_frame)
        frame = round(exact_frame)
        accumulated_error = exact_frame - frame

        # if accumulated_error >= 1.0:
        #     frame = math.ceil(exact_frame)
        #     accumulated_error -= 1.0
        # else:
        #     frame = math.floor(exact_frame)

        beat_to_frame_mapping.append(frame)

    return beat_to_frame_mapping


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


def animate(obj, d: OrderedDict):
    for prop, stamps in d.items():
        animate_property(obj, prop, stamps)


@bpy_handler
def main(C=None, D=None):
    seq = list(range(16))
    beat_unit = 16
    num_units = 4
    bpm = 120
    fps = 30
    beats = beat_sequence(seq=seq, beat_unit=beat_unit, num_units=num_units)
    frames = beats_to_frames(beats=beats, bpm=bpm, fps=fps)

    scale_seq = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)]
    rotate_seq = [(45, 45, 45), (90, 90, 90), (135, 135, 135), (180, 180, 180)]
    inter_scale_seq = copy.deepcopy(scale_seq)
    inter_rotate_seq = copy.deepcopy(rotate_seq)
    vals = cycle(seq=scale_seq, inter_seq=inter_scale_seq)
    vals_rotate = cycle(seq=rotate_seq, inter_seq=inter_rotate_seq)
    frames = expand_frames(frames=frames, width=len(vals[0]) // 2)
    final = sim_join(frames=collapse(frames), vals=collapse(vals))
    final_rotate = sim_join(frames=collapse(frames), vals=collapse(vals_rotate))
    print(frames)
    print(expand_shape(vals, len(frames)))
    print(final)

    cube = D.objects.get("Cube")
    delete_existing_keyframes(cube)
    animation_data = OrderedDict(
        [("scale", final), ("rotation_euler", final_rotate)]
    )
    animate(cube, animation_data)


if __name__ == "__main__":
    main()

import bpy
import math
import numpy as np


D = bpy.data
C = bpy.context

import bpy
from collections import OrderedDict


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


def animate_property(obj, prop, stamps):
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


if __name__ == "__main__":
    cube = D.objects.get("Cube")
    if cube:
        h = 10
        animation_data = OrderedDict(
            [
                (
                    "scale",
                    [
                        (0, (0, 0, 0)),
                        (100, (1, 1, 1)),
                        (200, (5, 5, 5)),
                        (300, (0, 0, 0)),
                    ],
                ),
                (
                    "location",
                    [
                        (0, (0, 0, 0)),
                        (100, (0, 0, h)),
                        (200, (0, h, h)),
                        (300, (0, h, 0)),
                    ],
                ),
            ]
        )
        animate(cube, animation_data)
    else:
        print("Cube object not found in the scene.")

import bpy
import math
import numpy as np


D = bpy.data
C = bpy.context

import bpy
from collections import OrderedDict
import copy


def beat_sequence(*, seq: list, beat_unit: int, num_units: int):
    res = copy.deepcopy(seq)
    shifted_vals = copy.deepcopy(seq)
    for _ in range(num_units - 1):
        shifted_vals = [shifted_vals[i] + beat_unit for i in range(len(seq))]
        res += shifted_vals
    return res


def map_beats_to_frames(beats, bpm, fps):
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
    # cube = D.objects.get("Cube")
    # if cube:
    #     h = 10
    #     animation_data = OrderedDict(
    #         [
    #             (
    #                 "scale",
    #                 [
    #                     (0, (0, 0, 0)),
    #                     (100, (1, 1, 1)),
    #                     (200, (5, 5, 5)),
    #                     (300, (0, 0, 0)),
    #                 ],
    #             ),
    #             (
    #                 "location",
    #                 [
    #                     (0, (0, 0, 0)),
    #                     (100, (0, 0, h)),
    #                     (200, (0, h, h)),
    #                     (300, (0, h, 0)),
    #                 ],
    #             ),
    #         ]
    #     )
    #     animate(cube, animation_data)
    # else:
    #     print("Cube object not found in the scene.")
    # vals = [0, 3, 5, 7, 9, 11, 13, 15]
    vals = list(range(16))
    beat_unit = 16
    num_units = 4
    bpm = 108
    fps = 24
    beats = beat_sequence(seq=vals, beat_unit=beat_unit, num_units=num_units)
    frames = map_beats_to_frames(beats=beats, bpm=bpm, fps=fps)
    print(beats)
    print(frames)
    print(np.diff(frames))

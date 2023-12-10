import bpy
import math
import numpy as np
from typing import Any
from collections import OrderedDict
import copy
from functools import wraps
from ..helpers import *


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

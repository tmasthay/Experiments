import bpy
import numpy as np
from math import radians
ah = bpy.data.texts['animation_helpers.py'].as_module()

def sync_rotation(theta, axis):
    cube = bpy.data.objects['Cube']
    cube.rotation_euler = (0,0,0)
    delta = tuple([theta*e for e in axis])

    frames_per_beat = 16
    d = int(frames_per_beat / 4)
    # spacing = list(d * np.array([
    #     3, 3, 3, 3, 3,
    #     2, 1, 3, 3, 3,
    #     3, 2
    # ]))
    spacing = list(d * np.array([
        2, 1,
        4,
        4, 1, 1,
        3, 2, 1,
        4,
        4, 1, 1,
        3
    ]))
    start_frame = 1036
    num_copies = 100
    def controller():
        def helper(obj, frame_no, i):
            obj.keyframe_insert(
                data_path="rotation_euler", 
                frame=(frame_no-1)
            )
            
            obj.rotation_euler = tuple([
                delta[j] + obj.rotation_euler[j] for j in range(3)
            ])
            obj.keyframe_insert(
                data_path="rotation_euler", 
                frame=frame_no
            )
            obj.keyframe_insert(
                data_path="rotation_euler",
                frame=(frame_no+1)
            )
        return helper
    
    ah.place_spaced_sequence(
        cube,
        spacing,
        start_frame,
        num_copies,
        controller()
    )

def build_rotate_cube():
    theta = 1.0
    axis = (0,0,1)
    sync_rotation(theta, axis)
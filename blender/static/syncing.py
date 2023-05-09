import bpy
import numpy as np

def delete_existing_keyframes(cylinder):
    if cylinder.animation_data:
        cylinder.animation_data_clear()

def get_scale(version):
    if( version == 0 ):
        return (0.025, 0.025, 1.0)
    elif( version == 1 ):
        return (0.025, 0.025, 5.005)
    
def place_blip(cylinder, frame_no):
    v1 = get_scale(0)
    v2 = get_scale(1)
    
    cylinder.scale = v1
    cylinder.keyframe_insert(data_path="scale", frame=(frame_no-1))
    cylinder.scale = v2
    cylinder.keyframe_insert(data_path="scale", frame=frame_no)
    cylinder.scale = v1
    cylinder.keyframe_insert(data_path="scale", frame=(frame_no+1))

    
def place_sequence(cylinder, the_sequence):
    for curr_frame in the_sequence:
        place_blip(cylinder, curr_frame)
        
def place_spaced_sequence(
        cylinder, 
        spacing, 
        start_frame,
        num_copies,
        overwrite=True
):
    if( overwrite ):
        delete_existing_keyframes(cylinder)
    w = [start_frame]
    for i in range(num_copies):
        for e in spacing:
            w.append(w[-1] + e)
    place_sequence(cylinder, w)

large_drum = {
    'spacing' : [11,11,11,12],
    'start' : 9
}

fast_drum_1 = {
    'spacing' : [
        22, 11,
        23, 11,
        23, 11, 22,
        12, 22, 11
    ],
    'start' : 12
}

fast_drum_2 = {
    'spacing' : [
        11, 22,
        12, 22, 11,
        23, 11,
        23
    ],
    'start' : 18
}

fast_drum_3 = {
    'spacing' : [
        11,
        23, 11, 22,
        12, 22, 11,
        23, 11
    ],
    'start' : 26
}

top_fast = bpy.data.objects['Cylinder.014']
middle_fast = bpy.data.objects['Cylinder.023']
bottom_fast = bpy.data.objects['Cylinder.024']
num_fast_copies = 100

place_spaced_sequence(
    top_fast,
    fast_drum_1['spacing'],
    fast_drum_1['start'],
    num_fast_copies,
    True
)

place_spaced_sequence(
    middle_fast,
    fast_drum_2['spacing'],
    fast_drum_2['start'],
    num_fast_copies,
    True
)

place_spaced_sequence(
    bottom_fast,
    fast_drum_3['spacing'],
    fast_drum_3['start'],
    num_fast_copies,
    True
)

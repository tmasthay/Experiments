import bpy
import numpy as np

def delete_existing_keyframes(cylinder):
    if cylinder.animation_data:
        cylinder.animation_data_clear()

def get_scale(scale):
    return (0.025, 0.025, scale)
    
def place_blip(cylinder, frame_no, scales):
    v1 = get_scale(scales[0])
    v2 = get_scale(scales[1])

    print(type(v1))
    
    cylinder.scale = v1
    cylinder.keyframe_insert(data_path="scale", frame=(frame_no-1))
    cylinder.scale = v2
    cylinder.keyframe_insert(data_path="scale", frame=frame_no)
    cylinder.scale = v1
    cylinder.keyframe_insert(data_path="scale", frame=(frame_no+1))

    
def place_sequence(cylinder, the_sequence, scales):
    for curr_frame in the_sequence:
        place_blip(cylinder, curr_frame, scales)
        
def place_spaced_sequence(
        cylinder, 
        spacing, 
        scales,
        start_frame,
        num_copies,
        overwrite=True
):
    if( overwrite ):
        print(str(cylinder))
        delete_existing_keyframes(cylinder)
    w = [start_frame]
    for i in range(num_copies):
        for e in spacing:
            w.append(w[-1] + e)
    place_sequence(cylinder, w, scales)







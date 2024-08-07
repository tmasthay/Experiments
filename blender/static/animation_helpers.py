import bpy
from mathutils import Vector

def delete_object(name):
    if name in bpy.data.objects:
        # If it does, delete it
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.delete()
        
def align(obj, pt=(0,0,0), axis=(0,0,1)):
    direction = Vector(pt) - obj.location
    direction.normalize()
    rot_diff = Vector(axis).rotation_difference(direction)
    obj.rotation_euler = rot_diff.to_euler()
    bpy.context.view_layer.update()

# Redraw the scene
bpy.context.view_layer.update()
def delete_existing_keyframes(obj):
    if obj.animation_data:
        obj.animation_data_clear()
    
# def place_blip(cylinder, frame_no, scales):
#     v1 = get_scale(scales[0])
#     v2 = get_scale(scales[1])
    
#     cylinder.scale = v1
#     cylinder.keyframe_insert(data_path="scale", frame=(frame_no-1))
#     cylinder.scale = v2
#     cylinder.keyframe_insert(data_path="scale", frame=frame_no)
#     cylinder.scale = v1
#     cylinder.keyframe_insert(data_path="scale", frame=(frame_no+1))
  
def place_sequence(obj, the_sequence, f):
    for (i,curr_frame) in enumerate(the_sequence):
        f(obj, curr_frame, i)
        
def place_spaced_sequence(
        obj, 
        spacing, 
        start_frame,
        num_copies,
        f,
        overwrite=True
):
    if( overwrite ): delete_existing_keyframes(obj)
    w = [start_frame]
    for i in range(num_copies):
        for e in spacing:
            w.append(w[-1] + e)
    place_sequence(obj, w, f)







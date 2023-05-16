import bpy

main_drum = bpy.data.texts['main_drum.py'].as_module()
fast_drum = bpy.data.texts['fast_drum.py'].as_module()
spinning_lasers = bpy.data.texts['spinning_lasers.py'].as_module()
rotate_cube = bpy.data.texts['rotate_cube.py'].as_module()

fast_drum.build_fast_drum()
main_drum.build_main_drum()
spinning_lasers.build_spinning_lasers()
rotate_cube.build_rotate_cube()
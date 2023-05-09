import bpy

main_drum = bpy.data.texts['main_drum.py'].as_module()
fast_drum = bpy.data.texts['fast_drum.py'].as_module()

fast_drum.build_fast_drum()
main_drum.build_main_drum()
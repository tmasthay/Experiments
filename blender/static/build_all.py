import os, sys, bpy

# Get the path to the directory containing your script files
script_directory = os.path.dirname(os.path.abspath(__file__))

# Add the directory to the Python path
if script_directory not in sys.path:
    sys.path.append(script_directory)

main_drum = bpy.data.texts['main_drum'].as_module()
fast_drum = bpy.data.texts['fast_drum'].as_module()

fast_drum.build_fast_drum()
main_drum.build_main_drum()
from animation_helpers import *
import bpy

# Define a function to create a cylinder with a given color and location
def create_colored_cylinder(name, color, location, scale):
    # Check if an object with the same name already exists
    if name in bpy.data.objects:
        # If it does, delete it
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.delete()

    # Create a new mesh object with the vertices and faces
    bpy.ops.mesh.primitive_cylinder_add(location=location, scale=scale)
    
    # Get the just created object
    cylinder = bpy.context.object

    # Rename it
    cylinder.name = name

    # Create a new material
    mat = bpy.data.materials.new(name + "_material")

    # Enable 'Use nodes':
    mat.use_nodes = True

    # Set the base color
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color

    # Assign it to the object
    cylinder.data.materials.append(mat)

def build_spinning_lasers():
    # Define the colors
    purple = (0.5, 0, 0.5, 1)
    lime_green = (0.5, 1, 0, 1)
    orange = (1, 0.5, 0, 1)

    # Create the cylinders
    create_colored_cylinder("top1", purple, (1, 0, 10), (0.025, 0.025, 20))
    create_colored_cylinder("top2", lime_green, (-1, 0, 10), (0.025, 0.025, 20))
    create_colored_cylinder("top3", orange, (0, 0, 10), (0.025, 0.025, 20))



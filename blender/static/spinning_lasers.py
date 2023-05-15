import bpy
import numpy as np
ah = bpy.data.texts['animation_helpers.py'].as_module()

# Define a function to create a cylinder with a given color and location
def create_colored_cylinder(**kw):
    name = kw['name']
    color = kw['color']
    scale = kw['scale']
    location = kw['location']
    emit = kw['emit']

    # Check if an object with the same name already exists
    if name in bpy.data.objects:
        # If it does, delete it
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.delete()

    # Create a new mesh object with the vertices and faces
    bpy.ops.mesh.primitive_cylinder_add(location=location)
    
    # Get the just created object
    cylinder = bpy.context.object

    # Scale the cylinder
    cylinder.scale = scale

    # Rename it
    cylinder.name = name

    # Create a new material
    mat = bpy.data.materials.new(name + "_material")

    # Enable 'Use nodes':
    mat.use_nodes = True

    # Get the material node tree
    nodes = mat.node_tree.nodes

    # Remove the default Principled BSDF node
    nodes.remove(nodes.get('Principled BSDF'))

    # Create an Emission node
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.name = "%s_emit"%name

    # Set the Emission color and strength
    emission_node.inputs['Color'].default_value = color
    emission_node.inputs['Strength'].default_value = emit

    # Create an Output node and connect the Emission node to it
    # output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node = nodes.get('Material Output')
    mat.node_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    # Assign it to the object
    cylinder.data.materials.append(mat)

def sync_spinning_lasers(colors, scale):
    frames_per_beat = 16
    d = int(frames_per_beat / 4)
    spacing = list(d * np.array([
        3, 3, 3, 3, 3,
        2, 1, 3, 3, 3,
        3, 2
    ]))
    start_frame = 532
    num_copies = 100
    def controller(phase):
        name = 'top%d'%(phase+1)
        emit_name = '%s_emit'%name
        def helper(obj, frame_no, i):
            obj.scale = (0.0, 0.0, 0.0)
            obj.keyframe_insert(data_path="scale", frame=(frame_no-1))
            
            obj.scale = scale
            obj.keyframe_insert(data_path="scale", frame=frame_no)
            
            emit_node = obj.data.materials[0].node_tree.nodes.get(emit_name)
            # emit_node = obj.data.get(emit_name)
            idx = np.mod((i + phase), len(colors))
            emit_node.inputs['Color'].default_value = colors[idx]
            emit_node.inputs['Color'].keyframe_insert(
                data_path="default_value", 
                frame=frame_no
            )
            
            obj.scale = (0.0,0.0,0.0)
            obj.keyframe_insert(data_path="scale", frame=(frame_no+1))
        
        return helper
    
    num_cylinders = 4
    for i in range(num_cylinders):
        ah.place_spaced_sequence(
            bpy.data.objects['top%d'%(i+1)],
            spacing,
            start_frame,
            num_copies,
            controller(i)
        )

def build_spinning_lasers():
    scale = (0.025, 0.025, 6.5)
    emit = 30.0
    align_point = (0,0,2)
    xy_perturb = 5.0
    height = 5.0

    colors = [
        (0.0, 0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 0.5),
        (0.0, 1.0, 1.0, 1.0)
    ]

    locations = [
        (-xy_perturb, 0.0, height),
        (xy_perturb, 0.0, height),
        (0.0, -xy_perturb, height),
        (0.0, xy_perturb, height)
    ]

    assert(len(locations) == len(colors))

    # Create the cylinders
    for i in range(len(colors)):
        name = 'top%d'%(i+1)
        create_colored_cylinder(
            name=name,
            color=colors[i],
            location=locations[i],
            scale=scale,
            emit=emit
        )
        ah.align(bpy.data.objects[name], align_point)
    sync_spinning_lasers(colors, scale)

# # Define a function to create a cylinder with a given color and location
# def create_colored_cylinder(**kw):
#     name = kw['name']
#     color = kw['color']
#     location = kw['location']
#     scale = kw['scale']
#     emit = kw['emit']

#     # Check if an object with the same name already exists
#     if name in bpy.data.objects:
#         # If it does, delete it
#         bpy.data.objects[name].select_set(True)
#         bpy.ops.object.delete()

#     # Create a new mesh object with the vertices and faces
#     bpy.ops.mesh.primitive_cylinder_add(location=location, scale=scale)
    
#     # Get the just created object
#     cylinder = bpy.context.object

#     # Rename it
#     cylinder.name = name

#     # Create a new material
#     mat = bpy.data.materials.new(name + "_material")

#     # Enable 'Use nodes':
#     mat.use_nodes = True

#     # Get the material node tree
#     nodes = mat.node_tree.nodes

#     # Remove the default Principled BSDF node
#     nodes.remove(nodes.get('Principled BSDF'))

#     # Create an Emission node
#     emission_node = nodes.new(type='ShaderNodeEmission')

#     # Set the Emission color and strength
#     emission_node.inputs['Color'].default_value = color
#     emission_node.inputs['Strength'].default_value = emit

#     # Create an Output node and connect the Emission node to it
#     output_node = nodes.new(type='ShaderNodeOutputMaterial')
#     mat.node_tree.links.new(
#         emission_node.outputs['Emission'], 
#         output_node.inputs['Surface']
#     )

#     # Assign it to the object
#     cylinder.data.materials.append(mat)
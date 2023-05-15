import bpy
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

    # Set the Emission color and strength
    emission_node.inputs['Color'].default_value = color
    emission_node.inputs['Strength'].default_value = emit

    # Create an Output node and connect the Emission node to it
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    # Assign it to the object
    cylinder.data.materials.append(mat)

def build_spinning_lasers():
    # Define the colors
    purple = (0.5, 0, 0.5, 1.0)
    yellow = (1.0, 1.0, 0.0, 1.0)
    orange = (1, 0.5, 0, 1)
    indigo = (0.29, 0.0, .51, 1.0)
    scale = (0.025, 0.025, 20)
    emit = 30.0
    align_point = (0,0,2)
    height = 5.0

    # Create the cylinders
    create_colored_cylinder(
        name="top1", 
        color=purple,
        location=(1.0, 0.0, height),
        scale=scale,
        emit=emit
    )
    create_colored_cylinder(
        name="top2", 
        color=yellow,
        location=(0.0, 1.0, height),
        scale=scale,
        emit=emit
    )
    create_colored_cylinder(
        name="top3", 
        color=orange,
        location=(1.0, 1.0, height),
        scale=scale,
        emit=emit
    )
    create_colored_cylinder(
        name="top4", 
        color=indigo,
        location=(0.0, 0.0, height),
        scale=scale,
        emit=emit
    )

    num_cylinders = 4
    for s in ['top%d'%i for i in range(1,num_cylinders+1)]:
        ah.align(bpy.data.objects[s], align_point)


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
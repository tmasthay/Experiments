import bpy
ah = bpy.data.texts['animation_helpers.py'].as_module()

def build_smoke():
    domain = 'smoke_domain'
    flow = 'smoke_flow'

    ah.delete_object(domain)
    ah.delete_object(flow)

    # Create a cube mesh for the smoke domain
    bpy.ops.mesh.primitive_cube_add(
        size=10, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, 0, 0)
    )
    domain_obj = bpy.context.object
    domain_obj.name = domain

    # Set up the smoke domain
    bpy.ops.object.modifier_add(type='SMOKE')
    domain_obj.modifiers["Smoke"].smoke_type = 'DOMAIN'

    # Create a smaller cube for the smoke flow object
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, 
        enter_editmode=False, 
        align='WORLD', 
        location=(0, 0, 0)
    )
    flow_obj = bpy.context.object
    flow_obj.name = flow

    # Set up the smoke flow
    bpy.ops.object.modifier_add(type='SMOKE')
    flow_obj.modifiers['Smoke'].smoke_type = 'FLOW'
    flow_obj.modifiers['Smoke'].flow_settings.smoke_color = (1, 0, 1)  

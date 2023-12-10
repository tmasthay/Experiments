import bpy
from . import helpers


class CanimateOperator(bpy.types.Operator):
    bl_idname = "object.canimate"
    bl_label = "canimate"

    def execute(self, context):
        print('canimate')
        print(type(helpers.animate))
        return {'FINISHED'}

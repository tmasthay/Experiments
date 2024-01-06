import bpy
import os
from . import helpers


class TanimateOperator(bpy.types.Operator):
    bl_idname = "object.tanimate"
    bl_label = "tanimate"

    keyframes: bpy.props.StringProperty(
        name="keyframes", description="keyframes"
    )
    overwrite: bpy.props.BoolProperty(name="overwrite", description="overwrite")

    def execute(self, context):
        if not self.keyframes.startswith(os.sep):
            root = os.path.abspath(
                os.path.join(os.environ['BLENDER_DEV_PATH'], 'animation_data')
            )
            self.keyframes = os.path.join(root, self.keyframes)
            if not self.keyframes.endswith('.pydict'):
                self.keyframes = os.path.join(
                    self.keyframes, 'keyframes.pydict'
                )

        with open(self.keyframes, 'r') as f:
            animation_data = eval(f.read())

        print(animation_data)

        for obj in bpy.context.selected_objects:
            if self.overwrite:
                helpers.delete_existing_keyframes(obj)
            helpers.animate(animation_data, obj=obj)
        return {'FINISHED'}

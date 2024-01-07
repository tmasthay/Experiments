import bpy
import os
from . import tanimate_helpers as th


class TanimateOperator(bpy.types.Operator):
    bl_idname = "object.tanimate"
    bl_label = "tanimate"

    keyframes: bpy.props.StringProperty(
        name="keyframes", description="keyframes"
    )
    overwrite: bpy.props.BoolProperty(name="overwrite", description="overwrite")
    path: bpy.props.StringProperty(
        description="path",
        default=(
            "/home/tyler/Documents/repos/Experiments/blender/animation_data"
        ),
    )

    def execute(self, context):
        if not self.keyframes.startswith(os.sep):
            self.keyframes = os.path.join(self.path, self.keyframes)
            if not self.keyframes.endswith('.pydict'):
                self.keyframes = os.path.join(
                    self.keyframes, 'keyframes.pydict'
                )

        with open(self.keyframes, 'r') as f:
            animation_data = eval(f.read())

        print(animation_data)

        for obj in bpy.context.selected_objects:
            if self.overwrite:
                th.delete_existing_keyframes(obj)
            th.animate(animation_data, obj=obj)
        return {'FINISHED'}

import bpy
import os
from . import helpers


class TanimateOperator(bpy.types.Operator):
    bl_idname = "object.tanimate"
    bl_label = "tanimate"

    keyframes: bpy.props.StringProperty(
        name="keyframes", description="keyframes"
    )

    def execute(self, context):
        print('tanimate')
        print(os.environ['ISL'])
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

        helpers.animate(animation_data, obj='active')
        return {'FINISHED'}

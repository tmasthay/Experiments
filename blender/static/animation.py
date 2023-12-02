import bpy
from collections import OrderedDict


def animate_property(obj, prop, stamps):
    """
    Animates a property of a Blender object over a series of time stamps.

    Args:
    obj (bpy.types.Object): The Blender object to animate.
    prop (str): The property of the object to animate (e.g., 'location', 'scale').
    stamps (list of tuples): A list of tuples where each tuple contains:
                             (frame_number, property_value)
                             property_value should be a tuple or list if the property is multi-dimensional.
    """
    # Ensure the object and property are valid
    if not hasattr(obj, prop):
        raise ValueError(f"The object does not have a property '{prop}'.")

    # Set keyframes for the specified property
    for frame_number, value in stamps:
        # Set the frame number
        bpy.context.scene.frame_set(frame_number)

        # Set the property value (ensuring it's the correct type)
        if isinstance(value, (int, float)):
            # Convert to tuple for single-dimension properties
            value = (value, value, value)
        setattr(obj, prop, value)

        # Insert a keyframe for the property at this frame
        obj.keyframe_insert(data_path=prop)


def animate(obj, d: OrderedDict):
    """
    Animates multiple properties of a Blender object based on a provided ordered dictionary.

    Args:
    obj (bpy.types.Object): The Blender object to animate.
    d (OrderedDict): An ordered dictionary where each key is a property name (e.g., 'location', 'scale')
                     and each value is a list of tuples for keyframes (frame_number, property_value).
    """
    for prop, stamps in d.items():
        animate_property(obj, prop, stamps)


# Example usage
if __name__ == "__main__":
    # Create a cube object in Blender before running this script
    cube = bpy.data.objects.get("Cube")
    if cube:
        h = 10
        # Define animation data as an OrderedDict
        animation_data = OrderedDict(
            [
                (
                    "scale",
                    [
                        (0, (0, 0, 0)),
                        (100, (1, 1, 1)),
                        (200, (5, 5, 5)),
                        (300, (0, 0, 0)),
                    ],
                ),
                (
                    "location",
                    [
                        (0, (0, 0, 0)),
                        (100, (0, 0, h)),
                        (200, (0, h, h)),
                        (300, (0, h, 0)),
                    ],
                ),
            ]
        )
        animate(cube, animation_data)
    else:
        print("Cube object not found in the scene.")

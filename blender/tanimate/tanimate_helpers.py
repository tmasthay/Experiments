from functools import wraps
import bpy


def none_handler(defaults):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for key, handler in defaults.items():
                if kwargs.get(key) is None:
                    kwargs[key] = handler()
            return func(*args, **kwargs)

        return wrapper

    return decorator


context_handler = none_handler({"C": lambda: bpy.context})
data_handler = none_handler({"D": lambda: bpy.data})
bpy_handler = none_handler({"C": lambda: bpy.context, "D": lambda: bpy.data})


def delete_existing_keyframes(obj):
    if obj.animation_data:
        obj.animation_data_clear()


@context_handler
def animate_property(obj, prop, stamps, C=None):
    if not hasattr(obj, prop):
        raise ValueError(f"The object does not have a property '{prop}'.")

    print(stamps)
    for frame_number, value in stamps:
        print(f'frame: {frame_number}, value: {value}')
        C.scene.frame_set(frame_number)

        # if isinstance(value, (int, float)):
        #     value = (value, value, value)
        setattr(obj, prop, value)

        obj.keyframe_insert(data_path=prop)


def animate(d: dict, *, obj="active"):
    if obj == "active":
        obj = bpy.context.active_object
    for prop, stamps in d.items():
        if isinstance(stamps, dict):
            if hasattr(obj, prop):
                animate(stamps, obj=getattr(obj, prop))
            else:
                animate(stamps, obj=obj[prop])
        else:
            animate_property(obj, prop, stamps)

from helpers import cycle, expand_frames, sim_join, easy_main, get_frames
import numpy as np
from omegaconf import DictConfig

# def main():
#     seq = list(range(32))
#     beat_unit = 32
#     num_units = 4
#     bpm = 128
#     fps = 24
#     beats = beat_sequence(seq=seq, beat_unit=beat_unit, num_units=num_units)
#     frames = beats_to_frames(beats=beats, bpm=bpm, fps=fps)
#     frames = list(np.array(frames) + 55)
#     scale_seq = [(1, 1, 1)]
#     inter_scale_seq = [(2, 2, 2)]
#     vals = cycle(seq=scale_seq, inter_seq=inter_scale_seq)
#     # input(vals)
#     frames = expand_frames(frames=frames, width=1)
#     final = sim_join(frames=collapse(frames), vals=collapse(vals))
#     animation_data = {"scale": final}
#     with open("keyframes.pydict", "w") as f:
#         f.write(str(animation_data))


# if __name__ == "__main__":
#     main()


def nested_dict(val, *, keys):
    if len(keys) == 1:
        return {keys[0]: val}
    else:
        return {keys[0]: nested_dict(val, keys=keys[1:])}


def nested_access(cfg: DictConfig, keys):
    if len(keys) == 1:
        return cfg[keys[0]]
    else:
        return nested_access(cfg[keys[0]], keys[1:])


@easy_main(get_frames)
def main(cfg: DictConfig):
    keys = [
        'anim',
        'active_material',
        'node_tree',
        'nodes',
        'Noise Texture',
        'inputs',
        'Distortion',
        'default_value',
    ]
    seq = nested_access(cfg, keys)
    final = sim_join(frames=cfg.frames.flatten(), vals=seq)
    final = [(u, v[0]) for u, v in final]
    with open('keyframes.pydict', 'w') as f:
        f.write(str(nested_dict(final, keys=keys[1:])))


if __name__ == "__main__":
    main()

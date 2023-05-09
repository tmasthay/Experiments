from scaling import *
import bpy

def build_fast_drum():
    top_fast = bpy.data.objects['Cylinder.014']
    middle_fast = bpy.data.objects['Cylinder.023']
    bottom_fast = bpy.data.objects['Cylinder.024']
    num_fast_copies = 100
    scales = [1.0, 5.005]

    d = 16
    fast_drum_1 = {
        'spacing' : [
            22, 11,
            23, 11,
            23, 11, 22,
            12, 22, 11
        ],
        'start' : int(1.25 * d)
    }

    fast_drum_2 = {
        'spacing' : [
            11, 22,
            12, 22, 11,
            23, 11,
            23
        ],
        'start' : int(1.75 * d)
    }

    fast_drum_3 = {
        'spacing' : [
            11,
            23, 11, 22,
            12, 22, 11,
            23, 11
        ],
        'start' : int(2.5 * d)
    }

    place_spaced_sequence(
        top_fast,
        fast_drum_1['spacing'],
        scales,
        fast_drum_1['start'],
        num_fast_copies,
        True
    )

    place_spaced_sequence(
        middle_fast,
        fast_drum_2['spacing'],
        scales,
        fast_drum_2['start'],
        num_fast_copies,
        True
    )

    place_spaced_sequence(
        bottom_fast,
        fast_drum_3['spacing'],
        scales,
        fast_drum_3['start'],
        num_fast_copies,
        True
    )

build_fast_drum()
from scaling import *

top_fast = bpy.data.objects['Cylinder.014']
middle_fast = bpy.data.objects['Cylinder.023']
bottom_fast = bpy.data.objects['Cylinder.024']
num_fast_copies = 100


fast_drum_1 = {
    'spacing' : [
        22, 11,
        23, 11,
        23, 11, 22,
        12, 22, 11
    ],
    'start' : 12
}

fast_drum_2 = {
    'spacing' : [
        11, 22,
        12, 22, 11,
        23, 11,
        23
    ],
    'start' : 18
}

fast_drum_3 = {
    'spacing' : [
        11,
        23, 11, 22,
        12, 22, 11,
        23, 11
    ],
    'start' : 26
}

place_spaced_sequence(
    top_fast,
    fast_drum_1['spacing'],
    fast_drum_1['start'],
    num_fast_copies,
    True
)

place_spaced_sequence(
    middle_fast,
    fast_drum_2['spacing'],
    fast_drum_2['start'],
    num_fast_copies,
    True
)

place_spaced_sequence(
    bottom_fast,
    fast_drum_3['spacing'],
    fast_drum_3['start'],
    num_fast_copies,
    True
)
import bpy

scaling = bpy.data.texts['scaling.py'].as_module()

def build_main_drum():
    delta = 16
    main_drum = {
        'spacing' : [4*delta],
        'start' : 16
    }

    drum1 = bpy.data.objects['Cylinder.004']
    drum2 = bpy.data.objects['Cylinder.001']
    drum3 = bpy.data.objects['Cylinder.018']
    drum4 = bpy.data.objects['Cylinder.003']
    num_copies = 100
    scales = [1.0, 44.0]

    scaling.place_spaced_sequence(
        drum1,
        main_drum['spacing'],
        scales,
        main_drum['start'],
        num_copies,
        True
    )

    scaling.place_spaced_sequence(
        drum2,
        main_drum['spacing'],
        scales,
        main_drum['start'] + delta,
        num_copies,
        True
    )

    scaling.place_spaced_sequence(
        drum3,
        main_drum['spacing'],
        scales,
        main_drum['start'] + 2 * delta,
        num_copies,
        True
    )

    scaling.place_spaced_sequence(
        drum4,
        main_drum['spacing'],
        scales,
        main_drum['start'] + 3 * delta,
        num_copies,
        True
    )

build_main_drum()
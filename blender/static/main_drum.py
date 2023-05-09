from scaling import *
import bpy

def build_main_drum():
    delta = 16
    main_drum = {
        'spacing' : [delta,delta,delta,delta+1],
        'start' : 9
    }

    drum1 = bpy.data.objects['Cylinder.004']
    drum2 = bpy.data.objects['Cylinder.001']
    drum3 = bpy.data.objects['Cylinder.009']
    drum4 = bpy.data.objects['Cylinder.003']
    num_copies = 100
    scales = [1.0, 44.0]

    place_spaced_sequence(
        drum1,
        main_drum['spacing'],
        scales,
        main_drum['start'],
        num_copies,
        True
    )

    place_spaced_sequence(
        drum2,
        main_drum['spacing'],
        scales,
        main_drum['start'] + delta,
        num_copies,
        True
    )

    place_spaced_sequence(
        drum3,
        main_drum['spacing'],
        scales,
        main_drum['start'] + 2 * delta,
        num_copies,
        True
    )

    place_spaced_sequence(
        drum3,
        main_drum['spacing'],
        scales,
        main_drum['start'] + 3 * delta,
        num_copies,
        True
    )

if( __name__ == "__main__" ):
    build_main_drum()
import simpleaudio as sa
import time
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import json
import os
import sys

def save_dict(my_dict, filename):
    filename = filename.replace('.json','') + '.json'
    with open(filename, "w") as f:
        json.dump(my_dict, f)
        
def read_dict(filename):
    filename = filename.replace('.json', '') + '.json'
    if( os.path.exists(filename) ):
        with open(filename, "r") as f:
            print('FOUND json "%s"!'%filename)
            return json.load(f)
    else:
        return None
        
def separate_audio(filename, num_comps=15):
    y, sr = librosa.load(filename)
    song_length = len(y) / sr
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    S_harmonic, _ = librosa.magphase(librosa.stft(y_harmonic))
    S_percussive, _ = librosa.magphase(librosa.stft(y_percussive))
    # S = np.abs(librosa.stft(y))
    comp, act = librosa.decompose.decompose(
        S_percussive, 
        n_components=num_comps, 
        max_iter=10)
    return S_harmonic, S_percussive, comp, act, song_length

def extract_data(filename, path=None, n=15):
    if( path == None ):
        path = filename.replace('.wav','.json')
    tmp = read_dict(path)
    if( tmp != None ):
        d = tmp
        d['harmonic'] = np.array(d['harmonic'])
        d['percuss'] = np.array(d['percuss'])
        d['comp'] = np.array(d['comp'])
        d['act'] = np.array(d['act'])
    else:
        Sh, Sp, comp, act, song_length = separate_audio(filename, n)
        d = {
            'harmonic': Sh.tolist(),
            'percuss': Sp.tolist(),
            'comp': comp.tolist(),
            'act': act.tolist(),
            'song_length': song_length }
        save_dict(d, path)
        d['harmonic'] = Sh
        d['percuss'] = Sp
        d['comp'] = comp
        d['act'] = act
    wave_obj = sa.WaveObject.from_wave_file(audio_file)
    d.update({'audio_file': wave_obj})
    return d

def visualize(filename, log_mode=True, scatter=True):
    dict_path = filename.replace('.wav', '')
    print('Extracting data...', end='', file=sys.stderr)
    num_comps = 100
    d = extract_data(filename, dict_path, num_comps)
    print('DONE!', file=sys.stderr)
    # freq = np.array(range(d['harmonic'].shape[0]))
    comps = np.array(range(num_comps))
    if( log_mode ):
        comps = np.log(1.0 + comps)
        d['harmonic'] = np.log(1.0 + d['harmonic'])
        d['percuss'] = np.log(1.0 + d['percuss'])
    num_frames = d['harmonic'].shape[1]
    fps = num_frames / d['song_length']
    delta_t = 1.0 / fps
    
    # cols = 3
    # rows = 5
    # cidx = lambda i,j : i*cols + j
    # f_size = (12,8)
    # fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=f_size)
    # ax1.set_ylim(0.0, np.max(d['harmonic']))
    # ax2.set_ylim(0.0, np.max(d['percuss']))
    # ax1.set_title('Harmonic')
    # ax2.set_title('Percussive')
    
    # idx = 0
    # curr_time = 0.0
    # prev_time = 0.0
    # plots = [[None for i in range(cols)] for j in range(rows)]
    # if( scatter ):
    #     scatter1 = ax1.scatter(freq, d['harmonic'][:,idx])
    #     scatter2 = ax2.scatter(freq, d['percuss'][:,idx])
    # else:
    #     line1, = ax1.plot(freq, d['harmonic'][:,idx])
    #     line2, = ax2.plot(freq, d['percuss'][:,idx])

    fig, ax = plt.subplots()
    ax.set_ylim(0.0, np.max(d['act']))
    ax.set_title('Activations: (%d, 0.0)')
    sp = ax.scatter(comps, d['act'][:,0])
    
    idx = 0
    curr_time = 0.0
    prev_time = 0.0
    
    fig.canvas.draw_idle()
    plt.pause(0.1)

    # Play the audio
    play_obj = d['audio_file'].play()

    start_time = time.time()
    # while play_obj.is_playing():
    #     # Get the current playback time in seconds
    #     curr_time = time.time() - start_time
    #     dt = curr_time - prev_time
    #     if( delta_t <= dt ):
    #         idx += 1
    #         plt.title('(%d,%f)'%(idx, curr_time))
    #         if( scatter ):
    #             scatter1.set_offsets(
    #                 np.column_stack((freq, d['harmonic'][:,idx])))
    #             scatter2.set_offsets(
    #                 np.column_stack((freq, d['harmonic'][:,idx])))
    #         else:
    #             line1.set_ydata(d['harmonic'][:,idx])
    #             line2.set_ydata(d['percuss'][:,idx])

    #         fig.canvas.draw_idle()
    #         plt.pause(0.00001)
    #         prev_time = time.time() - start_time
    while play_obj.is_playing():
        # Get the current playback time in seconds
        curr_time = time.time() - start_time
        dt = curr_time - prev_time
        if( delta_t <= dt ):
            idx += 1
            ax.set_title('Activations: (%d,%f)'%(idx, curr_time))
            sp.set_offsets(np.column_stack((comps, d['act'][:,idx])))

            fig.canvas.draw_idle()
            plt.pause(0.00001)
            prev_time = time.time() - start_time

        print(f"Current playback time: {curr_time:.2f} seconds", end='\r')

if( __name__ == "__main__" ):
    # Load the audio file
    audio_file = 'Requiem.wav'
    log_mode = False
    visualize(audio_file, log_mode=True)

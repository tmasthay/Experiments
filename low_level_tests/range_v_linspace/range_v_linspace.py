import numpy as np
import time
from float_range import float_range
from float_range_pure import float_range_pure
import sys
import matplotlib.pyplot as plt
import random
sys.path.append('..')
from typlotlib import *

def run_test(f, N, trials=1):
    versions = {
        'linspace': np.linspace(0,1,N), 
        'native_range': range(0,N), 
        'cython_range': float_range(0,1,N),
        'pure_range': float_range_pure(0,1,N)
    }
    t = {
        'linspace': 0.0,
        'native_range': 0.0,
        'cython_range': 0.0,
        'pure_range': 0.0
    }
    arr = {
        'linspace': np.empty(N),
        'native_range': np.empty(N), 
        'cython_range': np.empty(N),
        'pure_range': np.empty(N)
    }
    keys_random = list(versions.keys())
    random.shuffle(keys_random)
    for j in range(trials):
        for k in keys_random:
            i = 0
            top = time.time()
            for e in versions[k]:
                arr[k][i] = f(e)
                i += 1
            t[k] += time.time() - top
    for k in t.keys():
        t[k] /= trials
    return t, arr

def make_plots(N,t,filename='compare.pdf'):
    setup_gg_plot(
        fig_color='black',
        face_color='blue'
    )
    plot_art = set_color_plot_global(
        axis_color='white',
        leg_edge_color='white',
        leg_label_color='white',
        tick_color='white',
        xlabel=r'$N$',
        ylabel=r'$t$ (s)',
        use_legend=True,
        use_grid=False
    )
    plot_args = {
        'linspace': {
            'linestyle': '-',
            'color': 'red'
        },
        'native_range': {
            'linestyle': '-.',
            'color': 'blue'
        },
        'cython_range': {
            'linestyle': ':',
            'color': 'orange'
        },
        'pure_range': {
            'linestyle': ':',
            'marker': '*',
            'color': pre_colors[10]
        }
    }
    for k,v in t.items():
        plt.plot(N,v,label=k, **plot_args[k])
    plot_art()
    plt.savefig(filename)

def go():
    def f(x):
        return x * np.random.random()
    num_trials = 5
    N_vals = [2**i for i in range(2, 25)]
    t = {
        'linspace' : [],
        'native_range' : [],
        'cython_range' : [],
        'pure_range': []
    }
    for N in N_vals:
        u,arr = run_test(f, N, num_trials)
        print(int(np.log(N) / np.log(2)))
        for k in arr.keys():
            print('    %.8e'%random.choice(arr[k]))
        for (i,k) in enumerate(u.keys()):
            t[k].append(u[k])
    make_plots(N_vals,t)

if( __name__ == "__main__" ):
    go()
    



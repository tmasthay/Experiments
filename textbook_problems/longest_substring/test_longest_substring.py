import pytest
from longest_substring import longest_substring
import string
import time
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

def test_longest_substring_basic():
    s = "abcabcbb"
    assert longest_substring(s) == 3

def test_longest_substring_single_char():
    s = "bbbbb"
    assert longest_substring(s) == 1

def test_longest_substring_mixed():
    s = "pwwkew"
    assert longest_substring(s) == 3

def test_longest_substring_empty_string():
    s = ""
    assert longest_substring(s) == 0

def test_longest_substring_no_repeating_characters():
    s = "abcdefgh"
    assert longest_substring(s) == 8

def test_longest_substring_complexity():
    input_lengths = np.array(list(range(100, 200000, 1000)))
    execution_times = []

    for length in input_lengths:
        s = ''.join(random.choice(string.ascii_letters) for _ in range(length))
        t = time.time()
        v = longest_substring(s)
        execution_times.append(time.time() - t)
    
    #Get regression info
    p = np.polyfit(np.log(input_lengths), np.log(execution_times), 1)
    reg_execution = np.exp(p[1]) * input_lengths**p[0]

    filename = 'longest_substring_plots/performance.pdf'
    if( os.path.exists(filename) ):
        os.system('rm %s'%filename)

    palette = sns.color_palette("Set1", n_colors=3)
    palette = [plt.matplotlib.colors.to_rgb(color) for color in palette]
    plt.rcParams.update({'text.usetex' : True})
    plt.subplots(figsize=(10,8))
    plt.plot(
        input_lengths, 
        execution_times, 
        color=palette[0],
        linestyle='-',
        label='Actual Runtime'
    )
    plt.plot(
        input_lengths, 
        reg_execution, 
        color=palette[1],
        linestyle='-.',
        label='Regressed Runtime'
    )
    plt.xlabel('Input Length')
    plt.ylabel('Execution Time (s)')
    plt.title('Longest Substring Algorithm Complexity')
    plt.text(
        0.6,
        0.5,
        r'$m = %.4f \approx 1.0$'%p[0],
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    plt.legend()
    plt.savefig(filename)
    plot_gen = os.path.exists(filename)


    tol = 0.05
    slope = p[0]
    print(p)
    pass_test = np.abs(1.0 - slope) < tol
    assert pass_test, \
        "slope=%.2e, slope_pass=%s, plot_gen=%s"%(slope, pass_test, plot_gen)
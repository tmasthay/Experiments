import numpy as np
from itertools import chain, combinations

def get_indices(iterable):
    s = list(iterable)
    y = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return y 

def generate(level):
    if( level == -1 ):
        return lambda n : []
    max_index = int(level * (level+1) / 2)
    def helper(n):
        if( n == 0 ): return [level]
        else:
           subtractor = level
           k = n
           while( subtractor < k ):
               k -= subtractor
               subtractor -= 1
           prefix = list(np.flip(range(subtractor,level+1)))
           prefix.append(k-1)
           return prefix
    return helper

def generate_global(max_level):
    def helper(n):
        if( n == 0 ): return []
        if( n == 1 ): return [0]
        level = 1
        size = 2
        n = n - 2
        while( n >= size ):
            n -= size
            level += 1
            size = int(1 + level * (level+1) / 2)
        tmp = generate(level)
        return tmp(n)
    return helper

def generate_global_raw_int(max_level):
    u = generate_global(max_level)
    def helper(n):
        return sum([4**e for e in u(n)])
    return helper
 
max_level = 10
u = generate_global(max_level)
v = generate_global_raw_int(max_level)

final_seq_no = 100
for i in range(final_seq_no):
    print('%s --> %d'%(str(u(i)), v(i)))

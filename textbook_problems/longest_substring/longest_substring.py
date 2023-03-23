import numpy as np
        
def longest_substring(s):
    d = {}
    l = 0
    max_l = 0
    for ss in s:
        if( ss not in d.keys() ):
            d[ss] = True
            l += 1
            max_l = max(l, max_l)
        else:
            d = {ss: True}
            l = 1
    return max_l

if( __name__ == "__main__" ):
    s = 'pwwkew'
    print(longest_substring(s))
import numpy as np

def cumsum(x, inplace=False):
    if( inplace ):
        y = x
    else:
        y = np.copy(x)
    M = len(x)
    L = int(np.log(M) / np.log(2))
    N = 2**L
    tail = None if 2**L == M else np.cumsum(y[N:])
    for l in range(1, L+1):
        s,d = 2**l, 2**(l-1)
        for i in range(s-1, N, s):
            y[i] += y[i-d]
    for l in range(L-1,0,-1):
        s,d = 2**l, 2**(l-1)
        for i in range(s-1, N-d, s):
            y[i+d] += y[i]
    if( type(tail) != None ):
        y[N:] = tail + y[N-1]
    return y
            
if __name__ == "__main__":
    input_array = np.array([3, 1, 7, 0, 4, 1, 6, 3, 8.5,11.3])
    result = cumsum(input_array)
    print("Input array:", input_array)
    print("Prefix sum: ", result)

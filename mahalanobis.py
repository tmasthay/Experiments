import numpy as np

def mahalanobis(f):
    mean = sum(f) / len(f)
    variance = sum((f-mean)**2) / (len(f) - 1)
    def helper(x):
        if( type(x) in [float, int, float64] ):
            return (x-mean)**2/np.sqrt(variance)
        else:
            y = np.zeros(len(x))
            for (i,xx) in enumerate(x):
                y[i] = helper(xx)
            return y
    return helper

def mahalanobis_func(f,g):
    m1 = mahalanobis(f)
    m2 = mahalanobis(g)
    return m1(g), m2(f)

if( __name__ == "__main__" ):
    print('MAIN')
                
    

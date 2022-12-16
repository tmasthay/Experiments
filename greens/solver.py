import numpy as np

def make_accel_matrix(x,y,c=1.0):
    N = len(x)
    M = len(y)
    if( type(c) == float ):
        return c * np.ones(N*M)
    else:
        z = np.zeros(N*M)
        for j in range(M):
           for i in range(N):
               print('(%d,%d) --> %d'%(i,j,i+j*M))
               z[i + j*N] = c(x[i], y[j])
        return np.diag(z)

N = 8
M = 4
x = np.linspace(1,2,N)
y = np.linspace(1,2,M)
c = lambda x,y : np.exp(x)*y
A = make_accel_matrix(x,y,c)
print('\n'.join([' '.join([str(ee) for ee in e]) for e in A]))

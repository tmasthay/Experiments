import numpy as np
import matplotlib.pyplot as plt

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

def brownian_motion(mu, sig):
    def helper(N):
        x = np.zeros(N)
        x[0] = mu
        for i in range(1,N):
            x[i] = x[i-1] + np.random.normal(mu,sig)
        return x
    return helper

if( __name__ == "__main__" ):
    mu = 0.0
    sig = 1.0
    N = 100
   
    buffer = 1.5
    delta = buffer * sig * np.sqrt(N)
    a = mu - delta
    b = mu + delta
    x = np.linspace(a,b,N)
 
    generator = brownian_motion(mu,sig)
    walk1 = generator(N)
    walk2 = generator(N)
    
    m1,m2 = mahalanobis_func(walk1, walk2)

    plt.plot(x, walk1, label='Walk1')
    plt.plot(x, walk2, label='Walk2')
    plt.plot(x, m1, label='M12')
    plt.plot(x, m2, label='M21')
    plt.plot(x, m1+m2, label='Msum')
    plt.title('mu=%f, sig=%f'%(mu,sig))
    plt.savefig('walk.pdf')
                
    

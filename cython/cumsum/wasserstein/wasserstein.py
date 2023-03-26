from scipy.stats.sampling import NumericalInverseHermite
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import accumulate

def cumulative(u, prepend_zero=False):
    v = list(accumulate(u))
    if( prepend_zero ): v.insert(0,0)
    return np.array(v) / v[-1]

def quantile(U, p, dt, ot, div_tol=1e-16, tail_tol=1e-30):
    Q = np.zeros(len(p))
    q = 0
    t = np.where(U > 0)[0][0]
    for pp in p:
        if( t >= len(U) - 1 or pp > 1-tail_tol ):
            Q[q] = ot + (len(U) - 1) * dt
        elif( U[t] == 0.0 ): Q[q] = ot
        else:
            while( U[t+1] < pp and t < len(U) ): t += 1
            if( t == len(U) - 1 ): Q[q] = (len(U) - 1)*dt
            else:
                if( abs(U[t+1] - U[t]) >= div_tol ):
                    Q[q] = ot + dt * (t + (pp-U[t]) / (U[t+1] - U[t]))
                else:
                    Q[q] = ot + dt * t
        q += 1
    return Q

a = -5
b = 5
mu = 0.0
sig = 1.0
dist = norm(loc=mu, scale=sig)

N_base = int(1e3)
x_tmp = np.linspace(a,b,N_base)
class Dummy:
    def __init__(self, pdf_arg, cdf_arg):
        self.pdf = pdf_arg
        self.cdf = cdf_arg
        
t = time.time()
dummy = Dummy(
    interp1d(x_tmp, dist.pdf(x_tmp), kind='cubic'),
    interp1d(x_tmp, dist.cdf(x_tmp), kind='cubic'))
q = NumericalInverseHermite(dummy, domain=(a,b), order=3, u_resolution=1e-5)
setup_time = time.time() - t

N_vals = np.array([2**e for e in range(3,20)])
eval_times = np.empty(len(N_vals))
my_times = np.empty(len(N_vals))
l2 = np.empty(len(N_vals))
linf = np.empty(len(N_vals))
my_l2 = np.empty(len(N_vals))
my_inf = np.empty(len(N_vals))
eps = 1e-5
u = time.time()
for (i,N) in enumerate(N_vals):
    print('Iteration %d of %d...elapsed=%f'%(i,len(N_vals),time.time() - u))
    x = np.linspace(a,b,N)
    p = np.linspace(eps, 1-eps, N)
    dx = x[1] - x[0]
    t = time.time()
    y = q.ppf(p)
    eval_times[i] = time.time() - t
    t = time.time()
    my_y = quantile(dist.cdf(x), p, dx, a)
    my_times[i] = time.time() - t
    z = dist.ppf(p) - y
    my_z = dist.ppf(p) - my_y
    l2[i] = np.linalg.norm(z) / N
    linf[i] = np.max(z)
    my_l2[i] = np.linalg.norm(my_z) / N
    my_inf[i] = np.max(my_z)

plt.subplot(1,2,1)
plt.plot(N_vals, eval_times, label='Evaluation times')
plt.plot(N_vals, my_times, label='Mine')
plt.plot(N_vals, setup_time * np.ones(len(N_vals)), label='Setup time')
plt.xlabel('Problem size')
plt.ylabel('Time (s)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(N_vals, l2, linestyle=':', color='red', label='scipy L2')
plt.plot(N_vals, linf, linestyle='-.', color='red', label='scipy linf')
plt.plot(N_vals, my_l2, linestyle=':', color='blue', label='my l2')
plt.plot(N_vals, my_inf, linestyle='-.', color='blue', label='my inf')
plt.xlabel('Problem size')
plt.ylabel('Error')
plt.legend()
plt.show()
plt.close()


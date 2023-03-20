import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class Lagrangian(object):
    def __init__(self, **kw):
        self.dim = 1 if 'dim' not in kw.keys() else kw['dim']
        self.mc = np.zeros(self.dim) if 'mc' not in kw.keys() else kw['mc']
        self.uc = np.zeros(self.dim) if 'uc' not in kw.keys() else kw['uc']
        
        self.nc = 1 if 'nc' not in kw.keys() else kw['nc']
        self.pc = np.zeros(self.nc) if 'pc' not in kw.keys() else kw['pc']

    def L(self, u,m,p):
        raise NotImplementedError()
    def L_p(self, u,m,p):
        raise NotImplementedError()
    def L_u(self, u,m,p):
        raise NotImplementedError()
    def L_m(self, u,m,p):
        raise NotImplementedError()

class Mine(Lagrangian):
    def L(self, u,m,p):
        return np.dot(np.exp(m), np.cos(u)) + p * sum(u**5 - np.cos(m) - 1)
    def L_p(self, u, m, p):
        return u**5 - np.cos(m) - 1.0
    def L_u(self, u,m,p):
        return -np.exp(m) * np.sin(u) + 5.0 * p * u**4
    def L_m(self, u, m, p):
        self.uc = fsolve(lambda u_dummy : self.L_p(u_dummy, m, None), u)
        self.pc = fsolve(lambda p_dummy : self.L_u(self.uc, m, p_dummy), p)
        return np.exp(m) * np.cos(self.uc) + p * np.sin(m)

if( __name__ == "__main__" ):
    def direct_gradient_1d(m):
        return np.exp(m) * (np.cos( (1+ np.cos(m)) ** 0.2 ) \
            + 0.2 * np.sin( (1+np.cos(m)) ** 0.2 ) * np.sin(m) * ((1 + np.cos(m)) ** (-0.8)) )

    tmp = Mine(dim=1, uc=np.array([10.0]))
    m = np.linspace(-10,10,1000)
    g_adjoint_state = np.zeros(len(m))
    g_direct = np.zeros(len(m))
    for (i,mm) in enumerate(m):
        tau = 1.0
        u_start = tau * (1.0 - np.cos(mm))**(0.2)
        p_start = tau * np.exp(mm) * np.sin((1.0 - np.cos(mm))**(0.2)) / (5.0 * u_start**(0.8))
        g_adjoint_state[i] = tmp.L_m(u_start, np.array([mm]), p_start)
        g_direct[i] = direct_gradient_1d(mm)
    
    plt.title("Adjoint State Sanity Check")
    plt.plot(m, g_adjoint_state, label="Adjoint State Gradient")
    plt.plot(m, g_direct, label="Direct gradient")
    plt.legend()
    plt.savefig("adj_state_check.pdf")

    #print('(u,m,p, gradient) = (%s,%s,%s, %s)'%(tmp.uc, tmp.mc, tmp.pc, gradient_val))
    


        
    

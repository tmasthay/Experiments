import numpy as np
from tyler import *
from scipy.sparse import *
import time

class Solver:
    def __init__(self, **kw):
        self.l = kw.get('l', 1.0)
        self.mu = kw.get('mu', 1.0)
        self.rho = kw.get('rho', 1.0)
        self.nx = kw.get('nx', 100)
        self.ny = kw.get('ny', 100)
        self.ax = kw.get('ax', 0.0)
        self.bx = kw.get('bx', 1.0)
        self.ay = kw.get('ay', 0.0)
        self.by = kw.get('by', 1.0)
        self.nt = kw.get('nt', 1000) + 1
        self.dt = kw.get('dt', 0.001)
        self.f1 = kw.get('f1', lambda x,y,t : 0.0)
        self.f2 = kw.get('f2', lambda x,y,t : 0.0)
        self.spec_rad = kw.get('spec_rad', 0.75)
        self.T = (self.nt - 1) * self.dt
        self.dx = (self.bx - self.ax) / (self.nx - 1)
        self.dy = (self.by - self.ay) / (self.ny - 1)
        self.xi = (self.l + 2 * self.mu) / self.dx**2
        self.tau = (self.l + 2 * self.mu) / self.dy**2
        self.eta = (self.l + self.mu) / (4.0 * self.dx * self.dy)
        self.alpha4 = (2.0 * self.spec_rad - 1.0) / (self.spec_rad + 1.0)
        self.alpha3 = self.spec_rad / (self.spec_rad + 1.0)
        self.alpha2 = 0.25 * (1.0 - self.alpha4 + self.alpha3)**2
        self.alpha1 = 0.5 - self.alpha4 + self.alpha3
        self.eta = 2.0 / (self.dt**2 * (1.0-self.alpha2))
        self.gamma = -2.0 / (self.dt*(1.0-self.alpha2))
        self.theta = -self.alpha2 / (1.0 - self.alpha2)

    def __reset_calls(self, key):
        if( key in d.keys() ):
            d[key] = 0 

    def __create_mesh(self):
        self.t = np.linspace(0.0, self.T, self.nt)
        self.x = np.linspace(self.ax, self.bx, self.nx)
        self.y = np.linspace(self.ay, self.by, self.ny)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x, self.y)
        assert( self.x[1] - self.x[0] == self.dx )
        assert( self.y[1] - self.y[0] == self.dy )

    def __eval_spatial_field(self, u, field_name, csc=False):
        assert( type(field_name) == str )
        src_lambda = u if type(u) != str else \
            eval('self.%s'%(u.replace('self.','')))
        dest_array = lil_matrix((self.nx*self.ny, 1))
        for (j,yy) in enumerate(self.y):
            for (i,xx) in enumerate(self.x):
                dest_array[i + j * self.ny] = src_lambda(xx, yy)
        if( csc ):
            exec('self.%s = csc_array(dest_array)'%(field_name.replace('self.','')))
        else:
            exec('self.%s = dest_array'%(field_name.replace('self.','')))

    def __eval_temporal_spatial_field(self, u, field_name, csc=False):
        assert( type(field_name) == str )
        src_lambda = u if type(u) != str else \
            eval('self.%s'%(u.replace('self.','')))
        dest_array = lil_matrix((self.nt, self.nx*self.ny))
        for (k,tt) in enumerate(self.t):
            for (j,yy) in enumerate(self.y):
                for (i,xx) in enumerate(self.x):
                    dest_array[k,i + j * self.ny] = src_lambda(xx,yy,tt)
        if( csc ):
            exec('self.%s = csc_array(dest_array)'%(field_name.replace('self.','')))
        else:
            exec('self.%s = dest_array'%(field_name.replace('self.','')))
   
    def __build_matrices(self):
        n = min(self.nx,self.ny)
 
        if( self.nx != self.ny ):
            raise ValueError('Must have nx=ny for now, will support in future')

        pad = lambda x,n: np.concatenate([x,np.zeros(n)])
        pad1 = lambda v,m,n=0 : pad(v * np.ones(m),n)
     
        A = spdiags([pad1(1*self.xi,n), pad1(-2*self.xi,n), 
            pad1(self.xi,n)], 
            [-1,0,1], 
            self.nx,
            self.ny)

        B = spdiags([pad1(-self.eta,n), pad1(self.eta,n)],
            [-1,1],
            self.nx,
            self.ny)
        C = spdiags([pad1(-self.eta,n), pad1(self.eta,n)], 
            [1,-1],
            self.nx,
            self.ny)
    
        A2 = bmat([[A if i == j else None for j in range(self.ny)] for i in range(self.nx)])
        B2 = bmat(eval(symdiag([(n-1)*['B'],(n-1)*['C']], [-1,1])))
       
        self.stiff = bmat([[A2,B2],[self.dx**2/self.dy**2*B2,A2]])

        self.M = spdiags([self.rho * np.ones(self.stiff.shape[0])], [0])

        self.step_mat = self.eta * (1-self.alpha4) * self.M + (1.0 - self.alpha3) * self.stiff
        self.S1 = -self.eta * (1.0 - self.alpha4) * self.M + self.alpha3 * self.stiff
        self.S2 = self.gamma * (1.0 - self.alpha4) * self.M
        self.S3 = (self.alpha4 + self.theta * (1.0 - self.alpha4)) * self.M

    def __computeLU(self, method='COLAMD'):
        self.inv = linalg.splu(self.step_mat)

    def __build_rhs(self, time_step):
        talpha = (1-self.alpha3) * self.t[time_step] + \
            self.alpha3 * self.t[time_step-1]
        self.f1_curr = lambda x,y: self.f1(x,y,talpha)
        self.f2_curr = lambda x,y: self.f2(x,y,talpha)
        self.__eval_spatial_field('f1_curr', 'F1', csc=False)
        self.__eval_spatial_field('f2_curr', 'F2', csc=False)
        self.F = csc_array(vstack([self.F1,self.F2]))
        self.rhs = self.F \
            + self.S1 * self.disp_prev \
            + self.S2 * self.vel_prev \
            + self.S3 * self_accel_prev

    def __set_initial_condition(self):
        self.accel_prev = lil_matrix((2*self.nx*self.ny,1))
        self.vel_prev = lil_matrix((2*self.nx*self.ny,1))
        self.disp_prev = lil_matrix((2*self.nx*self.ny,1))
        self.disp = lil_matrix((2*self.nx*self.ny,1))
    
    @inc_obj('bound')
    def __setup(self):
        self.__create_mesh()
        self.__build_matrices()
        self.__set_initial_condition()
        self.__computeLU()

    def solve(self, time_step):
        t = time.time()
        self.__build_rhs(time_step)
        print(self.inv.solve(self.rhs))
        print(t - time.time())

if( __name__ == "__main__" ):
    u = Solver(nx=7,ny=7)
    u.solve(1)
    pretty_print(u.stiff.toarray(), blocks=[u.nx, u.ny])

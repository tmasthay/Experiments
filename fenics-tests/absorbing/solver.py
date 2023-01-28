from fenics import *
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import numpy as np

# Create a 2D square mesh with 5x5 elements
mesh = UnitSquareMesh(50,50)

# Define function space for displacement and stress
V = VectorFunctionSpace(mesh, 'P', 1)
W = FunctionSpace(mesh, 'P', 1)

# Define displacement, stress tensor and the gradient of displacement
u_trial = TrialFunction(V)
u_test = TestFunction(V)

#Initialize previous time steps
u = Function(V)
up = Function(V)
upp = Function(V)
up.assign(Constant((0.0,0.0)))
upp.assign(Constant((0.0,0.0)))

#Component dummy
ux = Function(W)
uy = Function(W)

#Initialize material properties
rho = Constant(1.0)
lmbda = Constant(1.0)
mu = Constant(1.0)
def sigma(u):
    return lmbda * tr(grad(u)) * Identity(2) + 2*mu*sym(grad(u))

# Define time step and end time
dt = 0.01
T = 1

# Define forcing term and boundary condition
sig_x = 0.1
sig_y = 0.1
sig_t = 1.0
amp = 1.0
x0 = 0.5
y0 = 0.5
t0 = 0.0
deg = 3
f_str = '%.8f * exp(-%.8f * pow((x[0]-%.8f), 2) - %.8f * %s'%(
    amp, 0.5 / sig_x**2, x0, 0.5 / sig_y**2,
    'pow((x[1]-%.8f),2))'%y0)
f_static = Expression((f_str, f_str), degree=deg)
f_static_evaled = project(f_static, V)
def f(t):
    return Constant(dt**2*(1-(t-t0)**2) * np.exp(-0.5 * ((t-t0) / sig_y)**2))
 
bc = DirichletBC(V, Constant((0, 0)), DomainBoundary())

# Define linear elasticity equations
a = rho*inner(u_trial, u_test)*dx
L_static = -Constant(dt**2)*inner(sigma(up), grad(u_test))*dx

A = assemble(a)
b_static = assemble(L_static)

fig, ax = plt.subplots()

N = int(np.round(T/dt)) + 1
global cb, cb_min, cb_max
cb_min = 0.0
cb_max = 0.0
def plot_step(i, obj, dynamic=True):
    global cb, cb_min, cb_max
    if( i > 0 and dynamic ):
        cb.remove()
    ux = project(dot(obj, Expression(('1.0', '0.0'), degree=1)), W)
    p = plot(ux)
    plt.title('Displacement at step %d, t=%f'%(i, i*dt))
    if( dynamic ):
        cb = fig.colorbar(p)
    elif( i == 0 ):
        y = np.load('colorbar.npy')
        cb = fig.colorbar(y[0])
    if( i == N - 1 ):
        np.save('colorbar.npy', [p])
    

def update(i, dynamic=True):
    if( i == 0 ):
        plot_step(0, upp)
    elif( i == 1 ):
        ax.clear()
        plot_step(1,up)
    else:
        ax.clear()
        t = i * dt
        print('t = %f'%t)
        tmp = time.time()
        L_tmp = inner(2.0*up - upp + f(t) * f_static_evaled,u_test)*dx
        b = b_static + assemble(L_tmp)
        bc.apply(A,b)
        tmp1 = time.time()
        print('Building rhs: %f seconds'%(tmp1 - tmp)) 
    
        solve(A, u.vector(), b)
        tmp2 = time.time()
        print('Solving system: %f seconds'%(tmp2 - tmp1))
    
        # Plot solution
        plot_step(i,u, dynamic)
        tmp3 = time.time()
        print('Plotting solution: %f seconds'%(tmp3 - tmp2))
    
        upp.assign(up)
        up.assign(u)
        tmp4 = time.time()
        print('Copying data: %f seconds'%(tmp4 - tmp3))
ani = FuncAnimation(fig, lambda i : update(i,True), frames=range(N), repeat=False)
ani.save('animation.gif', writer="imagemagick")


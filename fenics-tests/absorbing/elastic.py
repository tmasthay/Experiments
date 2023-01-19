from __future__ import print_function, absolute_import, division

import dolfin as dl
import math

import matplotlib.pyplot as plt


import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

n = 64
d = 1
mesh = dl.UnitSquareMesh(n, n)
Vh = dl.FunctionSpace(mesh, "Lagrange", d)
print("Number of dofs", Vh.dim())
dl.plot(mesh, title="FiniteElementMesh")
plt.savefig("1.pdf")

def boundary_d(x, on_boundary):
    return (x[1] < dl.DOLFIN_EPS or x[0] < dl.DOLFIN_EPS or x[0] > 1.0 - dl.DOLFIN_EPS) and on_boundary

u_d  = dl.Expression("sin(DOLFIN_PI*x[0])", degree = d+2)
bcs = [dl.DirichletBC(Vh, u_d, boundary_d)]

uh = dl.TrialFunction(Vh)
vh = dl.TestFunction(Vh)

f = dl.Constant(0.)
g = dl.Expression("DOLFIN_PI*exp(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[0])", degree=d+2)
a = dl.inner(dl.grad(uh), dl.grad(vh))*dl.dx
L = f*vh*dl.dx + g*vh*dl.ds

A, b = dl.assemble_system(a, L, bcs)
uh = dl.Function(Vh)
dl.solve(A, uh.vector(), b)

dl.plot(uh, title="FiniteElementSolution")
plt.savefig("2.pdf")

u_ex = dl.Expression("exp(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[0])", degree = d+2, domain=mesh)
grad_u_ex = dl.Expression( ("DOLFIN_PI*exp(DOLFIN_PI*x[1])*cos(DOLFIN_PI*x[0])",
                         "DOLFIN_PI*exp(DOLFIN_PI*x[1])*sin(DOLFIN_PI*x[0])"), degree = d+2, domain=mesh )

norm_u_ex   = math.sqrt(dl.assemble(u_ex**2*dl.dx))
norm_grad_ex = math.sqrt(dl.assemble(dl.inner(grad_u_ex, grad_u_ex)*dl.dx))

err_L2   = math.sqrt(dl.assemble((uh - u_ex)**2*dl.dx))
err_grad = math.sqrt(dl.assemble(dl.inner(dl.grad(uh) - grad_u_ex, dl.grad(uh) - grad_u_ex)*dl.dx))

print ("|| u_ex - u_h ||_L2 / || u_ex ||_L2 = ", err_L2/norm_u_ex)
print ("|| grad(u_ex - u_h)||_L2 / = || grad(u_ex)||_L2 ", err_grad/norm_grad_ex)

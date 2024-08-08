#!/usr/bin/env python

import sys
import math
import numpy as np
import m8r

par = m8r.Par()
inp = m8r.Input()
out = m8r.Output()

assert(inp.type == 'complex')

inp.settype('float')

# get dimensions
nw = inp.int('n1')
nx = inp.int('n2')

dw = inp.float('d1')
dx = inp.float('d2')

w0 = inp.float('o1')
x0 = inp.float('o2')

# get number of shots
n3 = inp.leftsize(2)

ctrace = np.zeros(2*nw,'f')

v1 = par.float('v1',0.0)
v2 = par.float('v2',0.1) # velocity gate

if v1>=v2:
    print("Need v1 < v2",file=sys.stderr) 

# Loop over shots
for i3 in range(n3):
    # Loop over wavenumber
    for ix in range(nx):
        x = math.fabs(x0 + ix*dx)+dx*sys.float_info.epsilon

        inp.read(ctrace)

	    # Loop over frequency
        for iw in range(nw):
            w = w0+iw*dw
            vel = w/x

            if vel>=-v2 and vel<=-v1:
                factor=1.0-math.sinf(0.5*math.pi*(vel+v2)/(v2-v1))
            elif vel>=-v1 and vel<=v1:
                factor=0.0 # reject 
            elif vel>=v1 and vel<=v2:
                factor=math.sin(0.5*math.pi*(vel-v1)/(v2-v1))
            else:
                factor=1.0 # pass
            # real and imaginary parts
            ctrace[2*iw]   *= factor
            ctrace[2*iw+1] *= factor

        out.write(ctrace)

sys.exit(0)
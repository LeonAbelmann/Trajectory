#**************************************************************************
# FieldSquare
# Calculate field components of a charged rectangle
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, sqrt
import scipy.integrate as integrate

import config

# **************************************************************************
# B(x,y,z,a,b)
# Returns Bx, By, Bz of rectangle in xy plane with size a x b,
# centered at (0,0).
# Charge density is 1, so multiply with magnetisation to get the actual
# magnetic field.
# Uses dblquad integration of scipy pacakge.
# https://
# docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html
# *************************************************************************
def Bsquare(x,y,z,a,b):
    # Note that in dblquad function is defined as func(y,x)!
    Bx = integrate.dblquad(dBx,-b/2,b/2,lambda y: -a/2,lambda y: a/2,
                            args=(x,y,z),epsabs=config.epsilon)
    if config.threed:
         By = integrate.dblquad(dBy,-b/2,b/2,lambda y: -a/2,lambda y: a/2,
                            args=(x,y,z),epsabs=config.epsilon)
    else:
        By = (0,0)
    Bz = integrate.dblquad(dBz,-b/2,b/2,lambda y: -a/2,lambda y: a/2,
                            args=(x,y,z),epsabs=config.epsilon)

    # Bx[0] is resulting integral, Bx[1] is error estimate
    # print("x=%3.3g, y=%3.3g, z=%3.3g, Bx=%g(%.1g), By=%g(%.1g), Bz=%g(%.1g) "
    #      % (x,y,z,Bx[0],Bx[1],By[0],By[1],Bz[0],Bz[1]))
    B = sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)

    C = 1/(4*pi)
    return (C*Bx[0],C*By[0],C*Bz[0])

#*************************************************************************
# dBx(xp,yp,x,y,z)
# Field of infinitely small area on square dA=dx*dy at (xp,yp)
#*************************************************************************
def dBx(xp,yp,x,y,z):
    sx = x-xp
    sy = y-yp
    sz = z
    s = sx**2 + sy**2 + sz**2
    dbx = sx/s**(3./2)
    return dbx

#*************************************************************************
# dBy(xp,yp,x,y,z)
# Field of infinitely small area on square dA=dx*dy at (xp,yp)
#*************************************************************************
def dBy(xp,yp,x,y,z):
    sx = x-xp
    sy = y-yp
    sz = z
    s = sx**2 + sy**2 + sz**2
    dby = sy/s**(3./2)
    return dby


#*************************************************************************
# dBz(xp,yp,x,y,z)
# Field of infinitely small area on square dA=dx*dy at (xp,yp)
#*************************************************************************
def dBz(xp,yp,x,y,z):
    sx = x-xp
    sy = y-yp
    sz = z
    s = sx**2 + sy**2 + sz**2
    dbz = sz/s**(3./2)
    return dbz

if 0:
    a = 1
    b = 1
    x = 9
    y = 10
    z = 11
    Bx,By,Bz = Bsquare(x,y,z,a,b)
    print("At (x,y,z)   : (%.3g, %.3g, %.3g)" % (x,y,z))
    print("B            : (%.3g, %.3g, %.3g)" % (Bx,By,Bz))

    # For testing, field of point charge (should match at r->infinity)
    r2 = x**2 + y**2 + z**2
    r = sqrt(r2)
    Btot = a*b/(4*pi*r2)
    print("Point charge : (%.3g, %.3g, %.3g)" % (Btot*x/r,Btot*y/r,Btot*z/r))


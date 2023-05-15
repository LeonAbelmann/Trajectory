#**************************************************************************
# FieldDisc
# Calculate field components of a charged disc
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, Aug 2022
#**************************************************************************
import numpy as np
from numpy import pi, sqrt, cos, sin
import scipy.integrate as integrate

import config

# **************************************************************************
# B(x,y,z,r)
# Returns Bx, By, Bz of disc in xy plane with radius r,
# centered at (0,0).
# Charge density is 1, so multiply with magnetisation to get the actual
# magnetic field.
# Uses dblquad integration of scipy pacakge.
# https://
# docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html
# *************************************************************************
def Bdisc(x,y,z,r):
    # Note that in dblquad function is defined as func(y,x)!
    Bx = integrate.dblquad(dBx,0,r,lambda y: 0,lambda y: 2*pi,
                            args=(x,y,z),epsabs=config.epsilon)
    if config.threed:
         By = integrate.dblquad(dBy,0,r,lambda y: 0,lambda y: 2*pi,
                            args=(x,y,z),epsabs=config.epsilon)
    else:
        By = (0,0)
    Bz = integrate.dblquad(dBz,0,r,lambda y: 0,lambda y: 2*pi,
                            args=(x,y,z),epsabs=config.epsilon)

    # Bx[0] is resulting integral, Bx[1] is error estimate
    # print("x=%3.3g, y=%3.3g, z=%3.3g, Bx=%g(%.1g), By=%g(%.1g), Bz=%g(%.1g) "
    #      % (x,y,z,Bx[0],Bx[1],By[0],By[1],Bz[0],Bz[1]))
    # B = sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)

    C = 1/(4*pi)
    return (C*Bx[0],C*By[0],C*Bz[0])

#*************************************************************************
# dBx(phi,r,x,y,z)
# Field of infinitely small area on disc dA=r*dr*dphi at (rp,phip)
#*************************************************************************
def dBx(phi,r, x,y,z):
    sx = x-r*cos(phi)
    sy = y-r*sin(phi)
    sz = z
    s = sx**2 + sy**2 + sz**2
    dbx = r*sx/s**(3./2)
    return dbx

#*************************************************************************
# dBy(phi,r,x,y,z)
# Field of infinitely small area on disc dA=r*dr*dphi at (rp,phip)
#*************************************************************************
def dBy(phi,r ,x,y,z):
    sx = x-r*cos(phi)
    sy = y-r*sin(phi)
    sz = z
    s = sx**2 + sy**2 + sz**2
    dby = r*sy/s**(3./2)
    return dby


#*************************************************************************
# dBz(phi,r,x,y,z)
# Field of infinitely small area on disc dA=r*dr*dphi at (rp,phip)
#*************************************************************************
def dBz(phi,r, x,y,z):
    sx = x-r*cos(phi)
    sy = y-r*sin(phi)
    sz = z
    s = sx**2 + sy**2 + sz**2
    dbz = r*sz/s**(3./2)
    return dbz

if 0:
    # Test to determine syntax of dblquad. Area of disc
    # Small area on disc dA = r dphi dr
    def dA(phi,r):
        return r

    R = 42 
    area=integrate.dblquad(dA,0,R,lambda y: 0,lambda y: 2*pi)
    area2 = pi*R**2
    print("Area = %f, should be %f" % (area[0],area2))

if 0:
    radius = 4.2
    x = 100*radius
    y = 0
    z = 110*radius
    Bx,By,Bz = Bdisc(x,y,z,radius)
    print("At (x,y,z)   : (%.3g, %.3g, %.3g)" % (x,y,z))
    print("B            : (%.3g, %.3g, %.3g)" % (Bx,By,Bz))

    # For testing, field of point charge (should match at r->infinity)
    r2 = x**2 + y**2 + z**2
    r = sqrt(r2)
    Btot = pi*radius**2/(4*pi*r2)
    print("Point charge : (%.3g, %.3g, %.3g)" % (Btot*x/r,Btot*y/r,Btot*z/r))
    print("")
    # very close to the disc, the field should be 1/2
    x=0
    y=0
    z=radius/1000
    Bx,By,Bz = Bdisc(x,y,z,radius)
    print("At (x,y,z)   : (%.3g, %.3g, %.3g)" % (x,y,z))
    print("B            : (%.3g, %.3g, %.3g)" % (Bx,By,Bz))
    print("Must be      : (%.3g, %.3g, %.3g)" % (0,0,0.5))
    

#*************************************************************************
# FieldCylinder
# Calculate field components and gradient of field of magnetised cylinder
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, cos, sin, sqrt

import config
import FieldDisc
from FieldDipole import Bdip, gradBdip

#for derivative
from scipy.misc import derivative

def norm(vec):
    vx,vy,vz = vec
    return sqrt(vx**2 + vy**2 +vz**2)

def norm2(vx,vy,vz):
    return (vx**2 + vy**2 +vz**2)

#*************************************************************************
# gradB(x,y,z)
# Returns gradient of B 
#*************************************************************************
def gradB(x0,y0,z0,r,h):
    def fx(x):
        return norm(B(x,y0,z0,r,h))
    def fy(y):
        return norm(B(x0,y,z0,r,h))
    def fz(z):
        return norm(B(x0,y0,z,r,h))
    dBdx = derivative(fx, x0, dx=config.DerP)
    if config.threed:
        dBdy = derivative(fy, y0, dx=config.DerP)
    else:
        dBdy = 0
    dBdz = derivative(fz, z0, dx=config.DerP)
    return (dBdx,dBdy,dBdz)

#*************************************************************************
# gradB2(x,y,z)
# Returns gradient of B^2 
#*************************************************************************
def gradB2(x0,y0,z0,a,b,h):
    # to be done
    # def fx2(x):
    #     return (B(x,y0,z0,a,b,h)[3])**2
    # def fy2(y):
    #     return (B(x0,y,z0,a,b,h)[3])**2
    # def fz2(z):
    #     return (B(x0,y0,z,a,b,h)[3])**2
    # dB2dx = derivative(fx2, x0, dx=config.DerP)
    # if config.threed:
    #     dB2dy = derivative(fy2, y0, dx=config.DerP)
    # else:
    #     dB2dy = 0
    # dB2dz = derivative(fz2, z0, dx=config.DerP)
    # return (dB2dx,dB2dy,dB2dz)
    return (0,0,0)


#*************************************************************************
# B(x,y,z,r,h)
# Field of cylinder, magnetised along z, with radius r and height h, centered
# at (0,0,0)
# Returns Bx, By, Bz
# Charge density is 1, so multiply with magnetisation to get the
# actual magnetic field.
#*************************************************************************
def B(x,y,z,r,h):
    # Approximate with dipole in far field
    # DAppr : how far away
    DAppr = config.DAppr
    r2=x**2+y**2+z**2
    if ((r2 > (DAppr*r)**2) and (r2 > (DAppr*h)**2)):
        config.NDip = config.NDip + 1 # NDip is global, see config.py
        #print("Dipole approximation, Ndip= %g" % config.NDip)
        return Bdip([x,y,z],pi*r**2*h,[0,0,1])
    Bf =  FieldDisc.Bdisc(x,y,z-h/2,r)
    Bb =  FieldDisc.Bdisc(x,y,z+h/2,r)
    Bx =  Bf[0] - Bb[0]
    By =  Bf[1] - Bb[1]
    Bz =  Bf[2] - Bb[2]
    return [Bx,By,Bz]

# Test
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
    # Magnet dimensions
    r=22.5
    h=30
    # Position
    x1 = 0
    y1 = 0
    z1 = 250
    # Do not approximate with dipole by setting cut-off distance big
    config.DAppr = 1e9
    # Reset number of dipole approximations made (to check)
    config.NDip = 0
    print("Magnet (r,h)     : (%.3g, %.3g)" % (r,h))
    print("Position (x,y,z) : (%.3g, %.3g, %.3g)" % (x1,y1,z1))
    # Calculate field
    Bx,By,Bz = B(x1,y1,z1,r,h)
    print("B                : (%.3g,%.3g,%.3g)"     % (Bx,By,Bz)) 

    # Calculate field of dipole for comparison (should match at r->inf)
    Bdipx,Bdipy,Bdipz = Bdip([x1,y1,z1],pi*r**2*h,[0,0,1])
    print("B dipole         : (%.3g,%.3g,%.3g)" % (Bdipx,Bdipy,Bdipz)) 

    # Same for gradients:
    gBx,gBy,gBz       = gradB(x1,y1,z1,r,h)
    print("gradB            : (%.3g,%.3g,%.3g)"% (gBx,gBy,gBz)) 
    gBdipx,gBdipy,gBdipz = gradBdip([x1,y1,z1],pi*r**2*h,[0,0,1])
    print("gradB dipole     : (%.3g,%.3g,%.3g)"% (gBdipx,gBdipy,gBdipz)) 

    print("Number of dipole approximations = %g" % config.NDip)


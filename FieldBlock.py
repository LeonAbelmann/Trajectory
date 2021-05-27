#**************************************************************************
# FieldBlock
# Calculate field components and gradient of field of magnetised block
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, cos, sin, sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
import FieldSquare
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
def gradB(x0,y0,z0,a,b,h,m):
    def fx(x):
        return norm(B(x,y0,z0,a,b,h,m))
    def fy(y):
        return norm(B(x0,y,z0,a,b,h,m))
    def fz(z):
        return norm(B(x0,y0,z,a,b,h,m))
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
# B(x,y,z,a,b,h,m)
# Field of block, magnetised along z, with dimensions a x b x h, centered
# at (0,0,0), with magnetisation components m=(mx, my, mz) (|m| = 1)
# Returns Bx, By, Bz
# Charge density is 1, so multiply with magnetisation to get the
# actual magnetic field.
#*************************************************************************
def B(x,y,z,a,b,h,m):
    mx, my, mz = m
    # Approximate with dipole in far field
    # DAppr : how far away
    DAppr = config.DAppr
    r=x**2+y**2+z**2
    if ((r > (DAppr*a)**2) and (r > (DAppr*b)**2) and (r > (DAppr*h)**2)):
        config.NDip = config.NDip + 1 # NDip is global, see config.py
        #print("Dipole approximation, Ndip= %g" % config.NDip)
        return Bdip([x,y,z],a*b*h,m)
    if mx !=0: # Rotate block around y +90 deg (x->-z, z->x)
        Bf =  FieldSquare.Bsquare(-z,y,x-a/2,h,b)
        Bb =  FieldSquare.Bsquare(-z,y,x+a/2,h,b)
        # Rotate fields back around y (-90) (x->z, z->-x)
        Bx =  mx*( Bf[2] - Bb[2])
        By =  mx*( Bf[1] - Bb[1])
        Bz =  mx*(-Bf[0] + Bb[0])
    if my !=0: # Rotate block around x +90 deg (y->-z, z->y)
        Bf =  FieldSquare.Bsquare(x,-z,y-b/2,a,h) 
        Bb =  FieldSquare.Bsquare(x,-z,y+b/2,a,h)
        # Rotate fields back around x (-90) (y->z, z->-y)
        Bx =  my*( Bf[0] - Bb[0])
        By =  my*( Bf[2] - Bb[2])
        Bz =  my*(-Bf[1] + Bb[1])
    if mz != 0:
        Bf =  FieldSquare.Bsquare(x,y,z-h/2,a,b)
        Bb =  FieldSquare.Bsquare(x,y,z+h/2,a,b)
        Bx =  mz*( Bf[0] - Bb[0])
        By =  mz*( Bf[1] - Bb[1])
        Bz =  mz*( Bf[2] - Bb[2])
    return [Bx,By,Bz]

# Test
if 0:
    # Magnet dimensions
    a=1
    b=2
    h=3
    # Magnetisation direction (|m|=1 )
    m = [1,0,0]
    # Position
    x1 = 9
    y1 = 10
    z1 = 11
    # Do not approximate with dipole by setting cut-off distance big
    config.DAppr = 1e9
    # Reset number of dipole approximations made (to check)
    config.NDip = 0
    print("Magnet (a,b,h)   : (%.3g, %.3g, %.3g)" % (a,b,h))
    print("Position (x,y,z) : (%.3g, %.3g, %.3g)" % (x1,y1,z1))
    # Calculate field
    Bx,By,Bz = B(x1,y1,z1,a,b,h,m)
    print("B                : (%.3g,%.3g,%.3g)"     % (Bx,By,Bz)) 

    # Calculate field of dipole for comparison (should match at r->inf)
    Bdipx,Bdipy,Bdipz = Bdip([x1,y1,z1],a*b*h,m)
    print("B dipole         : (%.3g,%.3g,%.3g)" % (Bdipx,Bdipy,Bdipz)) 

    # Same for gradients:
    gBx,gBy,gBz       = gradB(x1,y1,z1,a,b,h,m)
    print("gradB            : (%.3g,%.3g,%.3g)"     % (gBx,gBy,gBz)) 
    gBdipx,gBdipy,gBdipz = gradBdip([x1,y1,z1],a*b*h,m)
    print("gradB dipole     : (%.3g,%.3g,%.3g)"     % (gBdipx,gBdipy,gBdipz)) 

    print("Number of dipole approximations = %g" % config.NDip)


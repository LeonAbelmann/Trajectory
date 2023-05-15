#*************************************************************************
# FieldArray
# Calculate field of an array of magnets lying in the xy-plane.
# All magnets have equal width (y), but varying cross sections (x,z). 
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, cos, sin, sqrt

import config
import FieldBlock

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
#***********************************************************************
def gradB(x0,y0,z0, a = [1], b = 1, h = [1], m = [1,0,0]):
    def fx(x):
        return norm(B(x, y0, z0,
                          a=a, b=b, h=h, m=m))
    def fy(y):
        return norm(B(x0, y, z0,
                          a=a, b=b, h=h, m=m))
    def fz(z):
        return norm(B(x0, y0, z,
                          a=a, b=b, h=h, m=m))
    
    dBdx = derivative(fx, x0, dx=config.DerP)
    dBdy = derivative(fy, y0, dx=config.DerP)
    dBdz = derivative(fz, z0, dx=config.DerP)
    return (dBdx,dBdy,dBdz)

#************************************************************************
# B(x,y,z,a[],b,h[],m[])
# Field of Nx sets of of (z,-x,-z,x) magnetised magnets.
# x,y,z: coordinate where the field is calculated
# Size of magnets along (x,y,z) : (a[i],b,h[i])
# Magnetisation directions m[i] = (mx, my, mz)
# Top of magnets in xy plane, lower left corner is (0,0)
# Returns Bx, By, Bz
#************************************************************************
def B(x,y,z, a = [1],b = 1,h = [1],m = [1,0,0]):
    Bx = 0;
    By = 0;
    Bz = 0;
    shiftx = 0;
    for i in range(a.size):
        #print("i: %d, a=%g, b=%g, h=%g, m=(%g,%g,%g)" %
        #          (i,a[i],b,h[i],m[i,0],m[i,1],m[i,2]))

        # FieldBlock.B(x,y,z,a,b,h,m)
        # Field of block with dimensions a x b x h, centered
        # at (0,0,0), with magnetisation components mb=(mx, my, mz)
        # so you need to shift x a[i]/2 to the right, z h[i] down
        Bval = FieldBlock.B(x-(a[i]/2)-shiftx, y-b/2, z+(h[i]/2),
                                a[i], b, h[i], m[i])
        #print("B(%g, %g, %g)=(%g, %g, %g) " %
        #       (x-a[i]/2-shiftx,y-b/2,z+h[i]/2,Bval[0],Bval[1],Bval[2]))

        # every next magnet has to be placed to the right of all former
        # ones (shiftx)
        shiftx = shiftx + a[i]
        Bx = Bx + Bval[0]
        By = By + Bval[1]
        Bz = Bz + Bval[2]
    return (Bx,By,Bz)

# ####################################################################
# Output
# ####################################################################
tomm = 1000 # meter to mm
N = 4
a = np.zeros(N); h=np.zeros(N); m=np.zeros((N,3))
# Dimensions and number of magnets:
b    = 12e-3 # Same for all magnets
a[0] = 1e-3; h[0] = 1e-3; m[0] = [ 0, 0, 1]
a[1] = 1e-3; h[1] = 1e-3; m[1] = [-1, 0, 0]
a[2] = 1e-3; h[2] = 1e-3; m[2] = [ 0, 0,-1]
a[3] = 1e-3; h[3] = 1e-3; m[3] = [ 1, 0, 0]


# Test single point
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    # Location
    xpos = 4e-3
    ypos = b/2
    zpos = 0.1e-3
    
    Bx,By,Bz = B(xpos,ypos,zpos,
                a=a, b=b, h=h, m=m)
    gBx,gBy,gBz = gradB(xpos,ypos,zpos,
                a=a, b=b, h=h, m=m)
    print("At point %3.0g,%3.0g,%3.0g) :" % (xpos,ypos,zpos))
    print("B     : (%3.3g,%3.3g,%3.3g)" % (Bx,By,Bz))
    print("gradB : (%3.3g,%3.3g,%3.3g)" % (gBx,gBy,gBz))



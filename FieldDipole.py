#**************************************************************************
# FieldDipole
# Calculate field components and gradient of field of dipole
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, sqrt

#**************************************************************************
# gradBdip(x,y,z,vol)
# Returns gradient of B at position vec=(x,y,z) of dipole with magnetic
# moment 'moment' in Tesla and orientation m=mx,my,mz (|m| =1 )
#*************************************************************************
def gradBdip(vec,moment,m):
    x,y,z = vec
    mx, my, mz = m
    r=sqrt(x**2 + y**2 + z**2)
    xdBdx=0;ydBdx=0;zdBdx=0;
    xdBdy=0;ydBdy=0;zdBdy=0;
    xdBdz=0;ydBdz=0;zdBdz=0;
    if mx != 0:
        A = r**6*sqrt(z**2 + y**2 +4*x**2)
        xdBdz = -3*(z*(5*x**2 + z**2 + y**2))/A
        xdBdy = -3*(y*(5*x**2 + z**2 + y**2))/A
        xdBdx = -12*(x**3)/A
    if my != 0:
        A = r**6*sqrt(x**2 + z**2 +4*y**2)
        zdBdx = -3*(x*(5*y**2 + x**2 + z**2))/A
        zdBdz = -3*(z*(5*y**2 + x**2 + z**2))/A
        zdBdy = -12*(y**3)/A
    if mz != 0:
        A = r**6*sqrt(x**2 + y**2 +4*z**2)
        zdBdx = -3*(x*(5*z**2 + x**2 + y**2))/A
        zdBdy = -3*(y*(5*z**2 + x**2 + y**2))/A
        zdBdz = -12*(z**3)/A
    
    # print("pos:    [%g, %g, %g]" % (x,y,z))
    # print("m  :    [%g, %g, %g]" % (mx,my,mz))
    # print("mx : dB=[%g, %g, %g]" % (xdBdx,xdBdy,xdBdz))
    # print("my : dB=[%g, %g, %g]" % (ydBdx,ydBdy,ydBdz))
    # print("mz : dB=[%g, %g, %g]" % (zdBdx,zdBdy,zdBdz))

    C = moment/(4*pi)
    dBdx = C*(xdBdx+ydBdx+zdBdx)
    dBdy = C*(xdBdy+ydBdy+zdBdy)
    dBdz = C*(xdBdz+ydBdz+zdBdz)
    return [dBdx,dBdy,dBdz]

#*************************************************************************
# Bdip(x,y,z,m,vol)
# Magnetic of dipole with moment 'moment' in Tesla magnetized along m=(mx,my,mz)
# at position vec = (x,y,z)
# Returns Bx, By, Bz
#*************************************************************************
def Bdip(vec,moment,m):
    x,y,z = vec
    mx, my, mz = m
    r=sqrt(x**2 + y**2 + z**2)
    rx=x/r
    ry=y/r
    rz=z/r
    xBx=0;yBx=0;zBx=0;
    xBy=0;yBy=0;zBy=0;
    xBz=0;yBz=0;zBz=0;
    if mx != 0:
        xBz  = mx*3*rz*rx/r**3
        xBy  = mx*3*ry*rx/r**3
        xBx  = mx*(3*rx**2-1)/r**3
    if my != 0:
        yBx  = my*3*rx*ry/r**3
        yBz  = my*3*rz*ry/r**3
        yBy  = my*(3*ry**2-1)/r**3
    if mz != 0:
        zBx  = mz*3*rx*rz/r**3
        zBy  = mz*3*ry*rz/r**3
        zBz  = mz*(3*rz**2-1)/r**3
    
    # print("pos:    [%g, %g, %g]" % (x,y,z))
    # print("m  :    [%g, %g, %g]" % (mx,my,mz))
    # print("mx : B= [%g, %g, %g]" % (xBx,xBy,xBz))
    # print("my : B= [%g, %g, %g]" % (yBx,yBy,yBz))
    # print("mz : B= [%g, %g, %g]" % (zBx,zBy,zBz))
    
    C = moment/(4*pi)
    Bx = C*(xBx+yBx+zBx)
    By = C*(xBy+yBy+zBy)
    Bz = C*(xBz+yBz+zBz)
    return [Bx,By,Bz]

# Test
if 0:
    pos    = [0,0,1]
    m      = [0,0,1]
    moment = 1
    print("Bdip    : ", Bdip(pos,moment,m))
    print("gradBdip: ", gradBdip(pos,moment,m))


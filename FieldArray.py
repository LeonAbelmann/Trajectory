#*************************************************************************
# FieldArray
# Calculate field of an array of magnets lying in the xy-plane.
# All magnets have equal width (y), but varying cross sections (x,z). 
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, cos, sin, sqrt
import matplotlib.pyplot as plt

from multiprocessing import Process,Pool
import time

parallel = False # Paralllel processing cannot be used in interpreter

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
        print("i: %d, a=%g, b=%g, h=%g, m=(%g,%g,%g)" %
                  (i,a[i],b,h[i],m[i,0],m[i,1],m[i,2]))
        # FieldBlock.B(x,y,z,a,b,h,m)
        # Field of block with dimensions a x b x h, centered
        # at (0,0,0), with magnetisation components mb=(mx, my, mz)
        # so you need to shift x a[i]/2 to the right, z h[i] down
        Bval = FieldBlock.B(x-(a[i]/2)-shiftx, y-b/2, z+(h[i]/2),
                                a[i], b, h[i], m[i])
        print("B(%g, %g, %g)=(%g, %g, %g) " %
               (x-a[i]/2-shiftx,y-b/2,z+h[i]/2,Bval[0],Bval[1],Bval[2]))

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

# Statistics
starttime = time.time()
config.NDip = 0

# Test single point
if 1:
    # Location
    xpos = 4e-3
    ypos = b/2
    zpos = 0.1e-3
    
    Bx,By,Bz = B(xpos,ypos,zpos,
                a=a, b=b, h=h, m=m)
    gBx,gBy,gBz = gradB(xpos,ypos,zpos,
                a=a, b=b, h=h, m=m)
    print("Number of dipole approximations: %d" % config.NDip) 
    print("Total calculation time... : %g" % (time.time()-starttime))
    print("At point %3.0g,%3.0g,%3.0g) :" % (xpos,ypos,zpos))
    print("B     : (%3.3g,%3.3g,%3.3g)" % (Bx,By,Bz))
    print("gradB : (%3.3g,%3.3g,%3.3g)" % (gBx,gBy,gBz))

plot = 0
if 0:
    # Line plot along z
    calculate = 1
    plot = not calculate # First save calculated data to file, than plot
    csv_base_filename = "gradHal1D_z" # without .csv!
    xpos = 2e-3
    ypos = b/2
    zmin = 0.1e-3
    zmax = 4e-3
    N    = 40 # Number of points in graph

    # Define wrapper function for easy switch between
    # parallel/intepreter operation
    def fun(z):
        resx,resy,resz = gradB(xpos,ypos,z, a=a, b=b, h=h, m=m)
        # print("(% .3g,% .3g,% .3g) : (% .3g,% .3g,% .3g) " %
        #           (z,ypos,zpos,resx,resy,resz))
        return (resx, resy, resz)

    if calculate:    
        Z = np.linspace(zmin, zmax, num=N)
        U=Z.copy()
        V=U.copy()
        W=U.copy()
        
        if not parallel:
            for i in range(0,N):
                Z[i]=zmin+(zmax-zmin)*(i/N)
                result= fun(Z[i])
                U[i] = result[0]
                V[i] = result[1]
                W[i] = result[2]
                print("i=%2d (% .3g,% .3g,% .3g) : (% .3g,% .3g,% .3g) " %
                          (i,xpos,ypos,Z[i],U[i],V[i],W[i]))
            print("Number of dipole approximations: %d" % config.NDip)
            
        if parallel:
            # Create jobs, one per value of i
            index = range(N)
            
            def data_stream(a):
                    for i, av in enumerate(a):
                        yield (i), (Z[i]) 
                        
            # Create mapping onto fun, remember indices
            # and X,Y values
            # arg[0] = i
            # arg[1] = Z[i,j]
            def proxy(args):
                #print(args[0])
                return args[0], (args[1], fun(args[1]))
            
            if __name__ == "__main__":
                pool = Pool()
                results = pool.map(proxy, data_stream(index))
                pool.close()
                pool.join()
                
                for i , v in results:
                    U[i], V[i], W[i] = v[1]
                    print("(i=%2d (%.3g,%.3g,%.3g) (%.3g, %.3g, %.3g)"
                              % (i,xpos,ypos,Z[i],U[i],V[i],W[i]))

        print("Total calculation time... : %g" % (time.time()-starttime))
        np.savetxt(csv_base_filename+"_Z.csv", Z)
        np.savetxt(csv_base_filename+"_U.csv", U)
        np.savetxt(csv_base_filename+"_V.csv", V)
        np.savetxt(csv_base_filename+"_W.csv", W)

    #Plot line
    if plot:
        #csv_base_filename = "gradHalG_x" # without .csv!
        print(csv_base_filename+"_Z.csv")
        Zr = np.loadtxt(csv_base_filename+"_Z.csv")
        Ur = np.loadtxt(csv_base_filename+"_U.csv")
        Vr = np.loadtxt(csv_base_filename+"_V.csv")
        Wr = np.loadtxt(csv_base_filename+"_W.csv")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(Zr*tomm,Ur,'b', label = "grad(B)$_x$")
        plt.plot(Zr*tomm,Vr,'r', label = "grad(B)$_y$")
        plt.plot(Zr*tomm,Wr,'k', label = "grad(B)$_z$")
        plt.xlabel("Z / mm")
        plt.ylabel("grad(B)$_i$/ (B$_r$/m)")
        #plt.xlim(right=55)
        title = "x = %g mm, y= %g mm" % (xpos*tomm,ypos*tomm)
        plt.title(title)
        plt.rcParams["legend.loc"] = 'best'
        ax.legend()
        plt.rcParams['xtick.top']   = True
        plt.rcParams['ytick.right'] = True
        plt.axhline(linewidth=1, color='k')
        plt.axvline(linewidth=1, color='k')
        plt.tight_layout()
        plt.savefig(csv_base_filename+".pdf", format="pdf")
        plt.show()

# 2D image of fields in xz plane
if 0:
    calculate = 1
    plot = not calculate
    csv_base_filename = "Field2DHal_xz" # without .csv!
    
    # simulation area
    xmin = -1e-3 
    xmax =  5e-3
    ypos =  b/2
    zmin =  0.1e-3
    zmax =  4.1e-3
    gridx = 6*8
    gridz = 4*8

    if calculate:
        X = np.linspace(xmin, xmax, num=gridx)
        Z = np.linspace(zmin, zmax, num=gridz)
        X, Z = np.meshgrid(X, Z)
        U=X.copy()
        V=X.copy()
        W=X.copy()

        # Define wrapper function for easy switch between
        # parallel/intepreter
        # operation
        def fun(x, z):
            # Calculate B field:
            Bx, By, Bz = B(x,ypos,z,a=a, b=b, h=h, m=m)
            return (Bx, By, Bz)
            # or Gradient field, just outcomment one of either
            # gBx, gBy, gBz = gradB(x,ypos,z,a=a, b=b, h=h, m=m)
            # return (gBx, gBy, gBz)
    
        if (not parallel): 
            for i in range(gridz):
                for j in range(gridx):
                    U[i,j], V[i,j], W[i,j] = fun(X[i,j],Z[i,j])
                    print("(i,j)=(%.3d,%.3d)"
                              "(% .5f, % .5f) (% .5f, % .5f, % .5f)"
                        % (i,j,X[i,j],Z[i,j], U[i,j],V[i,j],W[i,j]))
        else:
            # Create jobs, one per set of (i,j)
            xindex = range(gridx)
            zindex = range(gridz)

            def data_stream(a, b):
                for i, av in enumerate(a):
                    for j, bv in enumerate(b):
                        yield (j,i), (X[j,i],Z[j,i]) # Cartesian indexing!

            # Create mapping onto fun, remember indices and X,Y values
            # arg[0] = (i,j)
            # arg[1] = (X[i,j],Z[i,j])
            def proxy(args):
                #print(args[0])
                return args[0], (args[1], fun(*args[1]))
        
            if __name__ == "__main__":
                pool = Pool()
                results = pool.map(proxy, data_stream(xindex, zindex))
            
                for k,v in results:
                    U[k], V[k], W[k] = v[1]
                    i, j = k
                    print("(i,j)=(%3d,%3d)"
                              "(% .5f, % .5f) (% .5f, % .5f, % .5f)"
                        % (i,j,X[i,j],Z[i,j], U[i,j],V[i,j],W[i,j]))
            
                pool.close()
                pool.join()
        if not parallel:
            print("Total number dipole approximations: %d " % config.NDip)
        print("Total calculation time : %g" % (time.time()-starttime))
        np.savetxt(csv_base_filename+"_X.csv", X)
        np.savetxt(csv_base_filename+"_Z.csv", Z)
        np.savetxt(csv_base_filename+"_U.csv", U)
        np.savetxt(csv_base_filename+"_V.csv", V)
        np.savetxt(csv_base_filename+"_W.csv", W)


    # Show plot 
    if plot:
        #csv_base_filename = "Field2DHal_xy" # without .csv!
        print(csv_base_filename+"_X.csv")
        Xr = np.loadtxt(csv_base_filename+"_X.csv")
        Zr = np.loadtxt(csv_base_filename+"_Z.csv")
        Ur = np.loadtxt(csv_base_filename+"_U.csv")
        Vr = np.loadtxt(csv_base_filename+"_V.csv")
        Wr = np.loadtxt(csv_base_filename+"_W.csv")

        nrows=len(Xr)
        ncols=len(Xr[0])
        print("nrows: % g, ncols: %g" % (nrows,ncols))
        norm = Xr.copy()
        for i in range(nrows):
            for j in range(ncols):
                norm[i,j] = sqrt( Ur[i,j]**2 + Vr[i,j]**2 + Wr[i,j]**2)
                Ur[i,j]    = Ur[i,j] / norm[i,j]
                Wr[i,j]    = Wr[i,j] / norm[i,j]
        ratio = (zmax-zmin)/(xmax-xmin)
        fig, ax = plt.subplots(figsize=(7, 7*ratio))

        c = ax.pcolormesh(Xr*tomm, Zr*tomm, norm, cmap='gray',
                            #vmin=0,
                            shading='gouraud')
        subsampx = 2 # subsample array, 1 is no sampling
        subsampy = 2 # subsample array, 1 is no sampling
        q = ax.quiver(  Xr[0::subsampx,0::subsampy]*tomm,
                        Zr[0::subsampx,0::subsampy]*tomm,
                        Ur[0::subsampx,0::subsampy],
                        Wr[0::subsampy,0::subsampy],
                    color = 'darkred',
                    scale_units='y', scale=3,
                    headwidth = 2.5, headlength = 3, headaxislength = 2.5)
        #vectorlength 1/scale * y axis length
        ax.set_aspect('equal')#, adjustable='box')
        plt.xlabel("X / mm")
        plt.ylabel("Z / mm")
        cbar = fig.colorbar(c, shrink=1)
        #cbar.ax.set_ylabel("B / (B$_r$)")
        cbar.ax.set_ylabel("grad(B) / (B$_r$/m)")
        title = "y = %g mm" % (ypos*tomm)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(csv_base_filename+".pdf", format="pdf")
        plt.show()


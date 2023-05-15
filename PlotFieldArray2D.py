#*************************************************************************
# PlotFieldArray2D
# Plot field of an array of magnets lying in the xy-plane.
# All magnets have equal width (y), but varying cross sections (x,z). 
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, cos, sin, sqrt
import matplotlib.pyplot as plt
from multiprocessing import Process,Pool
import time
import sys

from FieldArray import B,gradB
import config

# Settings
tomm = 1000 # meter to mm
N = 4
a = np.zeros(N); h=np.zeros(N); m=np.zeros((N,3))
# Dimensions and number of magnets:
b    = 12e-3 # Same for all magnets
a[0] = 1e-3; h[0] = 1e-3; m[0] = [ 0, 0, 1]
a[1] = 1e-3; h[1] = 1e-3; m[1] = [-1, 0, 0]
a[2] = 1e-3; h[2] = 1e-3; m[2] = [ 0, 0,-1]
a[3] = 1e-3; h[3] = 1e-3; m[3] = [ 1, 0, 0]

# simulation area
xmin = -1e-3 
xmax =  5e-3
ypos =  b/2
zmin =  0.1e-3
zmax =  4.1e-3
gridx = 6*2
gridz = 4*2

csv_base_filename = "Field2DHal_xz" # without .csv!
    
# First time when you run, or after changing parameters, set calculate to 1. Output files will be generated. You can subsequently show them by setting calculate to zero
calculate = 0

# End of settings

# Parallel processing if not in interpreter https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
if hasattr(sys, 'ps1'): 
    parallel = False
    print("Serial processing")
else:
    parallel = True
    print("Parallel processing")

# Statistics
starttime = time.time()
config.NDip = 0

# 2D image of fields in xz plane


plot = not calculate

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


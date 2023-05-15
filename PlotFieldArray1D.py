#*************************************************************************
# PlotFieldArray1D
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
N = 4
a = np.zeros(N); h=np.zeros(N); m=np.zeros((N,3))
# Dimensions and number of magnets:
b    = 12e-3 # Same for all magnets
a[0] = 1e-3; h[0] = 1e-3; m[0] = [ 0, 0, 1]
a[1] = 1e-3; h[1] = 1e-3; m[1] = [-1, 0, 0]
a[2] = 1e-3; h[2] = 1e-3; m[2] = [ 0, 0,-1]
a[3] = 1e-3; h[3] = 1e-3; m[3] = [ 1, 0, 0]

xpos = 2e-3
ypos = b/2
zmin = 0.1e-3
zmax = 4e-3
N    = 40 # Number of points in graph

csv_base_filename = "gradHal1D_z" # without .csv!

# Determine what is calculated
# Only one coordinate can be variable (var)
def fun(var):
    x = xpos
    y = ypos
    z = var
    # resx,resy,resz = B(x,y,z, a=a, b=b, h=h, m=m)
    resx,resy,resz = gradB(x,y,z, a=a, b=b, h=h, m=m)
    # print("(% .3g,% .3g,% .3g) : (% .3g,% .3g,% .3g) " %
    #           (z,ypos,zpos,resx,resy,resz))
    return (resx, resy, resz)

# First time when you run, or after changing parameters, set calculate to 1. Output files will be generated. You can subsequently show them by setting calculate to zero
calculate = 1
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

plot = not calculate # First save calculated data to file, than plot


if calculate:    
    Z = np.linspace(zmin, zmax, num=N)
    U = Z.copy()
    V = U.copy()
    W = U.copy()
        
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

# Plot line.
# Labels are not changed automatically: up to you
if plot:
    tomm = 1000 # meter to mm

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


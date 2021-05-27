#/usr/bin/python3
#*************************************************************************
# Trajectory
# Calculate trajectories in force fields consisting of magnetic and
# gravitational force
# Parameters for numerical approximation are listed in config.py
# Leon Abelmann, May 2021
#**************************************************************************
import numpy as np
from numpy import pi, cos, sin, sqrt 
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.patches as patches
import csv
import time
from multiprocessing import Process,Pool
from functools import partial
import multiprocessing.managers

# Settings
import config
parallel = True # Parallel processing. Cannot be used in interpreter

# Constants
mu0 = 4e-7*pi # Vacuum permeability
tomm = 1000 # m to mm

# Definition of Halbach array. See FieldHalbachArray for details.
from FieldArray import B, gradB

N = 10 # 10 4x4 magnets 
a = np.zeros(N); h=np.zeros(N); m=np.zeros((N,3))
b    = 12e-3 # Same for all magnets
a[0] = 4e-3; h[0] = 4e-3; m[0] = [ 0, 0, 1]
a[1] = 4e-3; h[1] = 4e-3; m[1] = [-1, 0, 0]
a[2] = 4e-3; h[2] = 4e-3; m[2] = [ 0, 0,-1]
a[3] = 4e-3; h[3] = 4e-3; m[3] = [ 1, 0, 0]
a[4] = 4e-3; h[4] = 4e-3; m[4] = [ 0, 0, 1]
a[5] = 4e-3; h[5] = 4e-3; m[5] = [-1, 0, 0]
a[6] = 4e-3; h[6] = 4e-3; m[6] = [ 0, 0,-1]
a[7] = 4e-3; h[7] = 4e-3; m[7] = [ 1, 0, 0]
a[8] = 4e-3; h[8] = 4e-3; m[8] = [ 0, 0, 1]
a[9] = 4e-3; h[9] = 4e-3; m[9] = [-1, 0, 0]
Hal_l  = 40e-3 # mm

# N = 32 # 16 sets of (1x1.5)+(1.5x1) magnets 
# a = np.zeros(N); h=np.zeros(N); m=np.zeros((N,3)) # [0..31]
# b    = 12e-3 # Same for all magnets
# for i in range (8): # [0...7]
#     a[0+4*i] = 1.0e-3; h[0+4*i] = 1.5e-3; m[0+4*i] = [ 0, 0, 1]
#     a[1+4*i] = 1.5e-3; h[1+4*i] = 1.0e-3; m[1+4*i] = [-1, 0, 0]
#     a[2+4*i] = 1.0e-3; h[2+4*i] = 1.5e-3; m[2+4*i] = [ 0, 0,-1]
#     a[3+4*i] = 1.5e-3; h[3+4*i] = 1.0e-3; m[3+4*i] = [ 1, 0, 0]
# Hal_l  = 40e-3 # mm

# N = 40 # 20 sets of (1x2.75)+(1x2) magnets 
# a = np.zeros(N); h=np.zeros(N); m=np.zeros((N,3)) # [0..39]
# b    = 12e-3 # Same for all magnets
# for i in range (10): # [0...9]
#     a[0+4*i] = 1.0e-3; h[0+4*i] = 2.75e-3; m[0+4*i] = [ 0, 0, 1]
#     a[1+4*i] = 1.0e-3; h[1+4*i] = 2.00e-3; m[1+4*i] = [-1, 0, 0]
#     a[2+4*i] = 1.0e-3; h[2+4*i] = 2.75e-3; m[2+4*i] = [ 0, 0,-1]
#     a[3+4*i] = 1.0e-3; h[3+4*i] = 2.00e-3; m[3+4*i] = [ 1, 0, 0]
# Hal_l  = 40e-3 # mm

Br     = 1.35  # Remanent magnetisation magnet in Tesla

# y coordinate of trajectory. We use symmetry (threed True in config.py), so make sure that y is center of array (b/2)
ypos = b/2

# Magnetic nanoparticle
Bp  = 0.1 # T, remanent moment
rp  = 170e-9 # m, radius of magnetic particle (m)
Vp  = (4/3)*pi*(rp)**3 # volume of single np (m^3)

# Channel and flow
eta = 1e-3 # Viscosity Pa.s (water 1e-3)
fvol = 1 # mL/min
channel_w = 5e-3   # channel width (m)
channel_h = 0.8e-3 # channel height (m)
channel_d = 0.2e-3 # separation between channel bottom and magnet (m)
xstart    = -5e-3  # where do we release the particles (0 is start of array)
vflow = fvol * ((1e-6)/60)/(channel_w * channel_h) # Average flow velocity m/s


# Cell
rc    = 5e-6 # Cell radius (m)
rho   = 0 # Density difference with water kg/m3
nump  = 10     # number of magnetic particles on cell
mp    = (Bp/mu0)*Vp*nump # Magnetic moment of cell
# mp    = 4e-15 # Override 
Vc    = (4/3)*pi*(rc)**3 # volume of cell [m3]
f     = 6*pi*rc*eta # friction coefficient (F/v)
vterm = rho*Vc*9.8/f # sinking velocity of cell


print("Array length (Hal_l)          : %g m"    % (Hal_l))
print("Starting point (xstart)       : %g m"    % (xstart))
print("Volume nanoparticle (Vp)      : %g m3"   % (Vp))
print("Cell volume (Vc)              : %g m3"   % (Vc))
print("Cell Magnetic moment (mp)     : %g Am2"  % (mp))
print("Cell friction coefficient (f) : %g Ns/m" % (f))
print("Cell drop velocity (vterm)    : %g m/s"  % (vterm))
print("Average flow velocity (vflow) : %g m/s"  % (vflow))

# Velocity field, defined in plane perpendicular to array (xz plane).
def velocity_field(t, xz):
    x, z = xz
    
    # Permanent dipole model F=m2*grad(B), 
    (gradx,grady,gradz)=gradB(x,ypos,z,a=a, b=b, h=h, m=m)
    # Force = mp*grad(B). velocity = F/f.
    # note that B is scaled for Br=1T, so multiply by Br
    prefactor = mp*Br/f
    # vflow is average flow, assume parabolic flow profile
    # Note that channel runs from channel_d to channel_d+channel_h,
    # so center is channel_d + channel_h/2
    # print("vflow : ", vflow)
    v_parabolic = 1.5*vflow*(1-((2*(z-channel_d)/channel_h)-1)**2)
    # print("flow at z = %g : %g" % (z,v_parabolic))
    vx = prefactor*gradx + v_parabolic
    vy = prefactor*grady
    vz = prefactor*gradz

    # Soft magnetic sphere model, chi>>3 F=(V2/mu0)*3*grad(B^2);
    # (gradx,grady,gradz)=gradB2design3(y,0,x)
    # prefactor = (V2/mu0)*3*Bscale**2
    # vx = prefactor*gradz
    # vy = prefactor*gradx - 9.8*rho*V2
    print("x = %.6f, z=%.6f, v=(%8.4e, %8.4e, %8.4e)" %
          (x,z,vx,vy,vz))
    return (vx, vz)

# Bottom: If the particles get hit the bottom
def bottom(t,xz):
    x,z = xz
    z0 = channel_d # lowest value of z below which we stop simulating
    if ((z<z0)):
        print("bottom: x: %g, z: %g" % (x,z))
    return (z-z0)
bottom.terminal = True
bottom.direction = -1

# Exit: If the particles leave the array
def exit(t,xz):
    x,z = xz
    x0 = Hal_l # end of the array, these particles we consider lost
    if ((x>x0)):
        print("exit: x: %g, z: %g" % (x,z))
    return (x-x0)
exit.terminal = True
exit.direction = 1

# Exit: If the particles reach entry of the array.
# For reverse trajectories
def entry(t,xz):
    x,z = xz
    x0 = xstart # Entry of channel, these particles are captured
    if ((x<x0)):
        print("entry: x: %g, z: %g" % (x,z))
    return (x-x0)
entry.terminal = True
entry.direction = -1

# Top: If the particles get hit the top.
# For reverse trajectories
def top(t,xz):
    x,z = xz
    # highest value of z above which we stop simulating:
    z0 = channel_d + channel_h 
    if ((z>z0)):
        print("top: x: %g, z: %g" % (x,z))
    return (z-z0)
top.terminal = True
top.direction = 1



# calcTrajectory: Simulate trajectory starting at 0,zstart.
# Write result to file
def calcTrajectory(zstart):
    xz0 = (xstart, zstart)
    csv_filename="T_{0:g}_{1:g}.csv".format(zstart*1000,fvol)
    t_span = (0, simulationtime)
    sol = solve_ivp(velocity_field, t_span, xz0,
                    method='RK23',
                    #first_step = simulationtime/100,
                    atol = 1e-5,
                    rtol = 2e-3,
                    max_step=simulationtime/config.Ntimesteps, 
                    events=(bottom,exit))
    # Write to file
    np.savetxt(csv_filename, sol.y)
    print("zstart: %g, Execution time: % g" %
          (zstart, time.time()-start))

# calcTrajectoryInv: Simulate time inverted trajectory starting
# at Hal_l,zstart. Write result to file
def calcTrajectoryInv(zstart):
    xz0 = (Hal_l, zstart)
    csv_filename="Tinv_{0:g}_{1:g}.csv".format(zstart*1000,fvol)
    t_span = (simulationtime,0) # backwards in time
    sol = solve_ivp(velocity_field, t_span, xz0,
                    method='RK23',
                    #first_step = simulationtime/100,
                    atol = 1e-5,
                    rtol = 2e-3,
                    max_step=simulationtime/config.Ntimesteps, 
                    events=(top,entry))
    # Write to file
    np.savetxt(csv_filename, sol.y)

    
# Check if trajectory hits bottom
def hit(solution):
    xend=solution[0][-1]
    zend=solution[1][-1]
    return (zend < channel_d + 0.01e-3)

# Calculate fraction of particles captures from maximums zstart
def calc_capture(zmax):
    rel = (zmax-channel_d)/(channel_h) # [0..1]
    efficiency = (rel)**2 * (3-2*rel) # [0..1]
    return efficiency

########################################################################
# Output
########################################################################

# Test velocityfield
if 1:
    x = 40e-3
    z = 0.6e-3
    # vflow = 0 # Switch off flow
    vx, vz = velocity_field(0,[x,z])
    vy = 0
    print("v = (% .2g, % .2g, % .2g) mm/s" % (vx*tomm,vy*tomm,vz*tomm))

# Vectorplot of velocityfield in xz plane
if 0:
    csv_base_filename = "velocityHal2D_xz" # without .csv!
    calculate = 1
    plot = not calculate # Calculate first, than plot from csv
    
    xmin =  -1e-3 # simulation area
    xmax =   41e-3
    zmin =   channel_d
    zmax =   channel_d + channel_h
    gridx = 6*4
    gridz = 2*4
    vflow = 0
    if calculate:
        X = np.linspace(xmin, xmax, num=gridx)
        Z = np.linspace(zmin, zmax, num=gridz)
        X, Z = np.meshgrid(X, Z)
        U=X.copy()
        V=U.copy()
        W=U.copy()
        
        # Define wrapper function for easy switch
        # between parallel/intepreter operation
        def fun(x, z):
            xz = x, z
            vx, vz = velocity_field(0, xz)
            return (vx, 0, vz)
    
        starttime = time.time()
        config.NDip = 0
        if (not parallel): 
            for i in range(gridz):
                for j in range(gridx):
                    U[i,j], V[i,j], W[i,j] = fun(X[i,j],Z[i,j])
                    print("(i,j)=(%.3d,%.3d) (% .5f, % .5f)"
                          " (% .5e, % .5e, % .5e)"
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
            # arg[1] = (X[i,j],Y[i,j])
            def proxy(args):
                print(args[0])
                return args[0], (args[1], fun(*args[1]))
        
            if __name__ == "__main__":
                pool = Pool(config.cores)
                results = pool.map(proxy, data_stream(xindex, zindex))
            
                for k,v in results:
                    U[k], V[k], W[k] = v[1]
                    i, j = k
                    print("(i,j)=(%3d,%3d) (% .5f, % .5f)"
                          " (% .5f, % .5f, % .5f)"
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


    # Show plot, if not calculate:
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
        #print("nrows: % g, ncols: %g" % (nrows,ncols))
        norm = Xr.copy()
        for i in range(nrows):
            for j in range(ncols):
                norm[i,j] = sqrt( Ur[i,j]**2 + Vr[i,j]**2 + Wr[i,j]**2)
                Ur[i,j]    = Ur[i,j] / norm[i,j]
                Vr[i,j]    = Vr[i,j] / norm[i,j]
                Wr[i,j]    = Wr[i,j] / norm[i,j]
        fig, ax = plt.subplots(figsize=(6,4))
        # In case you want correct aspect ratio:
        #ratio = (zmax-zmin)/(xmax-xmin)
        #fig, ax = plt.subplots(figsize=(7, 7*ratio))
        c = ax.pcolormesh(Xr*tomm, Zr*tomm, norm*1e6, cmap='gray',
                        #vmin=0,
                        shading='gouraud')
        subsampx = 1 # subsample array, 1 is no sampling
        subsampy = 1 # subsample array, 1 is no sampling
        q = ax.quiver(  Xr[0::subsampx,0::subsampy]*tomm,
                        Zr[0::subsampx,0::subsampy]*tomm,
                        Ur[0::subsampx,0::subsampy],
                        Wr[0::subsampy,0::subsampy],
                    color = 'darkred',
                    scale_units='x', scale=1,
                    headwidth = 2.5, headlength = 3, headaxislength = 2.5)
        #vectorlength 1/scale * y axis length
        #ax.set_aspect('equal')#, adjustable='box')
        plt.xlabel("X / mm")
        plt.ylabel("Z / mm")
        cbar = fig.colorbar(c, shrink=1)
        cbar.ax.set_ylabel("v (um/s)")
        title = "y = %g mm" % (ypos*tomm)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(csv_base_filename+".pdf", format="pdf")
        plt.show()

# Calculate trajectories
if 0:
    simulationtime=Hal_l/vflow # Average time for particle to pass array

    calculate = 1
    plot = not calculate # First calculate, than plot from csv file

    # Starting point of trajectory
    startpoints = [0.82e-3]
    # Other examples:
    #startpoints =  np.arange(0.3e-3,1.1e-3,0.1e-3)
    #startpoints = [0.72e-3,0.74e-3,0.76e-3,0.78e-6]
    
    if calculate:
        start = time.time()
        if (not parallel): 
            for z0 in startpoints:
                calcTrajectory(z0)
        else:
            if __name__ == '__main__':
                with Pool(config.cores) as p:
                    p.map(calcTrajectory, startpoints)
    
    # Load and plot csv file
    if plot :
        minhit = 0
        maxhit = Hal_l
        import glob
        allfiles=glob.glob("T_*.csv")
        print(allfiles)
        fig, ax = plt.subplots(1)
        for fname in allfiles:
            sol=np.loadtxt(fname)
            # check if trajectory hit magnet
            if hit(sol):
                xcoor = sol[0][0]
                if (xcoor < minhit):
                    minhit = xcoor
                if (xcoor > maxhit):
                    maxhit = xcoor
                #print("Hit at x0= %g" % xcoor)
                # Make that curve red
                curvecolor = 'r'
            else:
                curvecolor = 'k'
            plt.plot(sol[0]*1000, sol[1]*1000, curvecolor);
        # Plot graph
        plt.xlim(xstart, Hal_l*tomm)
        plt.ylim(channel_d*tomm, (channel_d+channel_h)*tomm)
        plt.xlabel('x / mm'); plt.ylabel('z / mm');
        plt.title("%gx%g,%gx%g, %g ml/min" %
                  (a1*tomm,h1*tomm,a2*tomm,h2*tomm,fvol))
        plt.savefig("Trajectory.pdf", format="pdf")
        plt.show();    

# Calculate maximum zstart which stil is captured
if 0:
    # Average time for particle to pass array:
    simulationtime=(Hal_l-xstart)/vflow 

    calculate = 1
    plot = not calculate # First calculate, than plot from csv file

    if calculate:
        start = time.time()
        # Start calculating backwards from capture point
        # (Parallel does not make sense, there is only one trajectory)
        calcTrajectoryInv(channel_d)
    # Load and plot csv file
    if plot :
        csv_filename="Tinv_{0:g}_{1:g}.csv".format(channel_d*1000,fvol)
        print(csv_filename)
        sol=np.loadtxt(csv_filename)
        # z coordinate of last simulated point is particle with highest
        # zstart that still reaches the array
        zend=sol[1][-1]
        capture=calc_capture(zend)
        print("zend    : %g mm" % (zend*tomm))
        print("Capture : %g" % (capture))

        # Plot trajectory
        fig, ax = plt.subplots(1)
        plt.plot(sol[0]*1000, sol[1]*1000, color='k');
        plt.xlim(xstart, Hal_l*tomm)
        plt.ylim(channel_d*tomm, (channel_d+channel_h)*tomm)
        plt.xlabel('x / mm'); plt.ylabel('z / mm');
        plt.title("%gx%g,%gx%g, %g ml/min, %g fAm$^2$" %
                  (a[0]*tomm,h[0]*tomm,a[1]*tomm,h[1]*tomm,fvol,mp*1e15))
        ax.text(0.6*Hal_l*tomm, (channel_d+0.8*(channel_h))*tomm,
                "Start      : %.3g mm \nCapture : %.3g" %
                (zend*tomm,capture), fontsize=12)
        plt.savefig("TrajectoryInv.pdf", format="pdf")
        plt.show(); 

#**************************************************************************
# config
# Global parameters to tune numerical approximation
# Leon Abelmann, May 2021
#**************************************************************************

# If false, ignore the y component of the fields and gradients to speed up calculation in case of xz-plane symmetry
threed = False

# Epsilon, precision of integration. See
# https:
# //docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html
# Absolute tolerance passed directly to the inner 1-D quadrature integration. Default is 1.49e-8. dblquad` tries to obtain an accuracy of abs(i-result) <= max(epsabs, epsrel*abs(i)) where i = inner integral of func(y, x) from gfun(x) to hfun(x), and result is the numerical approximation. See epsrel below.
epsilon = 0.1

# DAppr: Dipole approximation for r**2 > factor * largest of (a,d,h)**2
DAppr = 3

# NDip: Keep track of the number of times a dipole approximation was made
NDip = 0

# DerP: Step used for differential (m). Small is more accurate, but too
# small leads to noise
DerP = 1e-5

# Number of calculations per trajectory is simulationtime/Ntimesteps
Ntimesteps = 50

# Number of cores you want to use for parallel processing
cores = 4 

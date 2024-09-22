import numpy as np
import scipy as sp
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
import matplotlib.pyplot as plt
import generatemats
from quadrules import quadrules
from math import gamma as gamma_func
import pickle
from numba import jit, njit
import sys
import time

######## PARAMETERS ########
p 		= 3 						# Solution polynomial approximation order
nElems 	= 128  						# Number of spatial elements
Nx 		= 2**6+1 						# Number of discretization points in velocity space
M 		= 1.6 						# Mach number
Kn 		= 1.0						# Knudsen number
tf 		= 0.8						# Final time
CFL 	= 0.8 						# CFL number
niters 	= 2 						# Number of iterations for discrete velocity model
timescheme = ['euler', 'rk4'][1] 	# Time integration scheme 
dom = [0., 50.] 					# Spatial domai nextent
############################

gamma = 3.

# Compute initial macroscopic conditions
def IC(x):
	# Density, velocity, pressure
	U = np.zeros((npts, 3))

	# Unit density/pressure at inlet 
	[rl, pl] = [1., 1.]

	# Calculate speed of sound and set velocity based on Mach number
	cl = np.sqrt(gamma*pl/rl)
	ul = M*cl
	
	# Calculate outlet state from Rankine-Hugoniot conditions
	rr = rl*((gamma + 1)*M**2)/((gamma - 1)*M**2 + 2)
	ur = ul*((gamma - 1)*M**2 + 2)/((gamma + 1)*M**2)
	pr = pl*(2*gamma*M**2 - (gamma-1))/(gamma + 1)
	
	# Apply some initial smoothing to the shock profile to accelerate convergence
	x0 = 0.5*(dom[0] + dom[1])
	c1 = 0.5*(np.tanh(-25*(x-x0)) + 1)

	U[:,0] = c1*(rl) + (1-c1)*(rr)
	U[:,1] = c1*(ul) + (1-c1)*(ur)
	U[:,2] = c1*(pl) + (1-c1)*(pr)

	return U

# Compute analytic equilibrium distribution function
@jit
def makeEquilibriumDistribution(u, alpha):
	theta = 1.0/(2.*alpha[1])

	# Compute Maxwellian for velocity space
	M = alpha[0]*np.exp(-alpha[1]*((u - alpha[2])**2))

	return M

# Compute discretely conservative equilibrium distribution function using
# the discrete velocity model
@njit
def setEquilibriumDistribution(U, u, cinvs):
	# Get primitives
	r = U[:,0]
	V = U[:,1]
	P = U[:,2]

	# Get conserved variables
	Ucon = np.zeros_like(U)
	Ucon = PriToCon(U, Ucon)

	# Initialize distribution function
	g = np.zeros((npts, Nx))

	for i in range(npts):
		# Set initial guess as analytic distribution function
		theta = P[i]/r[i]
		alpha = np.zeros(3)
		alpha[0] = r[i]/(2*np.pi*theta)**(1./2.)
		alpha[1] = 1.0/(2*theta)
		alpha[2] = V[i]
		h = makeEquilibriumDistribution(u, alpha)

		# Take Newton iterations to find discretely conservative equilibrium distribution function
		for it in range(niters):
			uu = np.zeros(3)
			uh = getU1D(h, uu, u, cinvs)
			F = uh - Ucon[i, :]
			J = getJacobian(h, alpha)
			alpha = alpha - np.linalg.solve(J, F)
			h = makeEquilibriumDistribution(u, alpha)

		g[i,:] = h

	return g

# Pre-compute collision invariants
def getCollisionInvariants(u):
	cinvs = np.zeros((len(u), 3))
	cinvs[:, 0] = 1.
	cinvs[:, 1] = u
	cinvs[:, 2] = 0.5*(u**2)

	return cinvs

# Compute macroscopic state from distribution function (at one point)
@njit
def getU1D(f, U, z, cinvs):
	for var in range(3):
		U[var] = np.dot(cinvs[:,var]*f, phasewts)
	return U

# Compute macroscopic state from distribution function
@njit
def getU(f, U, z, cinvs):
	for var in range(3):
		U[..., var] = np.dot(cinvs[...,var]*f, phasewts)
	return U

# Compute Jacobian for Newton solver for discrete velocity model
@njit
def getJacobian(f, alpha):
	global u, cinvs
	J = np.zeros((3, 3))

	for var in range(3):
		dvar = 1./alpha[0]
		fu = dvar*cinvs[:,var]*f
		J[var, 0] = np.dot(fu, phasewts)

		dvar = -(u-alpha[2])**2
		fu = dvar*cinvs[:,var]*f
		J[var, 1] = np.dot(fu, phasewts)

		dvar = 2*alpha[1]*(u-alpha[2])
		fu = dvar*cinvs[:,var]*f
		J[var, 2] = np.dot(fu, phasewts)

	return J

# Compute time derivative for distribution function
@njit
def dfdt(f, t, u, cinvs):
	global bc_fl, bc_fr

	# Compute upwind divergence 
	divF = np.zeros_like(f)
	c1 = u > 0
	divF = u * (c1*(DL @ f) + (1-c1)*(DR @ f))

	# Add boundary condition contributions
	fl = f[0,:]
	dfl = c1*(bc_fl - fl)
	dfl *= u
	divF[:p+1, :] += np.outer(leftcorrGrad, dfl)/detJ

	fr = f[-1,:]
	dfr = (1-c1)*(bc_fr - fr)
	dfr *= u
	divF[-p-1:, :] += np.outer(rightcorrGrad, dfr)/detJ

	# Compute BGK source term
	Ucon = np.zeros((npts, 3))
	Ucon = getU(f, Ucon, u, cinvs)

	U = np.zeros((npts, 3))
	U = ConToPri(Ucon, U)
	M = setEquilibriumDistribution(U, u, cinvs)

	# Set collision time adaptively to recover temperature-based viscosity law
	omg = 0.81 # Viscosity exponent
	rho = U[:,0]
	P = U[:,2]
	theta = P/rho
	tau = tau_ref*(rho_ref*theta_ref**(1-omg))/(rho*theta**(1-omg))

	Q = ((M - f).T/tau).T # BGK source term

	return Q - divF

# Compute time derivative for distribution function by Boltzmann integral
@njit
def dfdt_boltzmann(f, t, z, cinvs):
    global bc_fl, bc_fr
    
	# Compute upwind divergence
    divF = np.zeros_like(f)
    c1 = u > 0
    divF = u * (c1*(DL @ f) + (1-c1)*(DR @ f))

	# Add boundary condition contributions
    fl = f[0,:]
    dfl = c1*(bc_fl - fl)
    dfl *= u
    divF[:p+1, :] += np.outer(leftcorrGrad, dfl)/detJ

    fr = f[-1,:]
    dfr = (1-c1)*(bc_fr - fr)
    dfr *= u
    divF[-p-1:, :] += np.outer(rightcorrGrad, dfr)/detJ

    
    Q = np.zeros_like(f)
    for pt in range(0, len(f[:,0])):
        Q[pt,:] = collisionInt(f[pt,:])

    return Q, divF

@njit
def collisionInt(f):
    n = len(f)
    N = (n-1)//2
    freq = fftshift(fftfreq(n))
    freq = freq*N/np.max(freq)
    
    T = np.pi
    R = 2*T/(1+3*np.sqrt(2))*2
    def phi2(s, R):
        	return 2*R*np.sinc(R*s)
    
    f_hat = fftshift(fft(f))
    Q_hat = np.zeros(n)
    for k in range(n):
        if k <= N:
            f_hat_l = f_hat[:N+k+1]
            l = freq[:N+k+1]

            f_hat_m = f_hat[N+k::-1]
            m = freq[N+k::-1]

			#print((l + m) - k)
        else:
            f_hat_l = f_hat[k-N:]
            l = freq[k-N:]

            f_hat_m = f_hat[k-N:][::-1]
            m = freq[k-N:][::-1]

			# print((l + m) - k)

        beta_lm = np.pi/2*(phi2(l, R) + phi2(-m, R))
        Q_hat[k] = np.sum(np.real(beta_lm*f_hat_l*f_hat_m))


    Q = np.real(ifft(ifftshift(Q_hat)))
    Q = np.roll(Q, -1)
    
    return Q

# Limit distribution function to > 0 using Zhang-Shu squeeze limiter
@njit
def limit(f):
	if p == 0:
		return f

	tol = 1e-12

	stride = p+1
	for i in range(nElems):
		ff = f[i*stride:(i+1)*stride,:]

		# Loop through velocity/internal energy space
		for j in range(Nx):
			# Get minimum value and element-wise spatial average
			fmin = np.min(ff[:,j])
			favg = np.dot(basiswts/2.0, ff[:,j])
			# If minimum falls below bounds, apply squeeze limiter
			if fmin < tol:
				zeta = (tol - fmin)/np.maximum(favg - fmin, tol)
				ff[:,j] = (1-zeta)*ff[:,j] + zeta*favg

		f[i*stride:(i+1)*stride,:] = ff

	return f

# Compute appropriate min/max values for discrete velocity domain
def getVelocityBounds(U):
	# Compute average macroscopic velocity and its max variation in the domain
	uu = U[:, 1]
	umax = np.max(uu)
	umin = np.min(uu)
	u0 = 0.5*(umax + umin)
	du = umax - umin

	# Set bounds as a factor of the thermal velocity (3-5 is standard here)
	k = 3
	thetamax = np.max(U[:,-1]/U[:,0])
	
	# Add velocity offsets
	ubounds = [u0 - k*thetamax - du, u0 + k*thetamax + du]

	print('Velocity bounds 		:', ubounds)
	return ubounds

# Setup discretization for velocity space
def setupVelocitySpace(N):
	global ubounds

	# Uniform discretization in velocity space
	pts = np.linspace(ubounds[0], ubounds[1], N)

	# Trapezoidal rule for integration
	wts = np.ones_like(pts)
	wts[0] = 0.5
	wts[-1] = 0.5
	wts /= np.sum(wts)
	wts *= ubounds[1] - ubounds[0]

	return [pts, wts]

# Setup discretization for internal energy space
def setupInternalSpace(N, U):
	global ubounds, velspace

	epsfac = 15.175 # See Table 1 in Dzanic et al. "A positivity-preserving and conservative high-order flux
	# reconstruction method for the polyatomic Boltzmann–BGK equation" for this value
	thetamax = np.max(U[:,-1]/U[:,0])
	L = epsfac*thetamax

	# Uniform discretization in internal energy space (remove point at eps = 0 as it's undefined there)
	pts = np.linspace(0, L, N+1)[1:]

	# Trapezoidal rule for integration
	wts = np.ones_like(pts)
	wts[0] = 0.5
	wts[-1] = 0.5
	wts /= np.sum(wts)
	wts *= L

	print('Internal energy bounds	:', [0, L])

	return [pts, wts]

# Compute next time step
@njit
def step(f_time, f_transport, t, dt, u, cinvs):
    global timescheme
    if timescheme == 'rk4':
        [Q, divF] = dfdt_boltzmann(f_time+f_transport, t, u, cinvs)
        f_time_k1, f_transport_k1 = Q*dt, -divF*dt
        
        rk4_1 = f_time + f_time_k1/2.
        ffac = np.sum(rk4_1, axis=1)/f0
        for i in range(0, len(f_time)):
            rk4_1[i] /= ffac[i]
        rk4_1 += f_transport + f_transport_k1/2.
        
        [Q, divF] = dfdt_boltzmann(limit(rk4_1), t + dt/2., u, cinvs)
        f_time_k2, f_transport_k2 = Q*dt, -divF*dt
        
        rk4_2 = f_time + f_time_k2/2.
        ffac = np.sum(rk4_2, axis=1)/f0
        for i in range(0, len(f_time)):
            rk4_2[i] /= ffac[i]
        rk4_2 += f_transport + f_transport_k2/2.
        
        [Q, divF] = dfdt_boltzmann(limit(rk4_2), t + dt/2., u, cinvs)
        f_time_k3, f_transport_k3 = Q*dt, -divF*dt
        
        rk4_3 = f_time + f_time_k3
        ffac = np.sum(rk4_3, axis=1)/f0
        for i in range(0, len(f_time)):
            rk4_3[i] /= ffac[i]
        rk4_3 += f_transport + f_transport_k3
        
        [Q, divF] = dfdt_boltzmann(limit(rk4_3), t + dt, u, cinvs)
        f_time_k4, f_transport_k4 = Q*dt, -divF*dt
        
        f_time = f_time + f_time_k1/6. + f_time_k2/3. + f_time_k3/3. + f_time_k4/6.
        ffac = np.sum(f_time, axis=1)/f0
        for i in range(0, len(f_time)):
            f_time[i] /= ffac[i]
        f_transport = f_transport + f_transport_k1/6. + f_transport_k2/3. + f_transport_k3/3. + f_transport_k4/6.
        return f_time, f_transport
    if timescheme == 'euler':
           [Q, divF] = dfdt_boltzmann(f_time+f_transport, t, u, cinvs)
           f_time = f_time + Q*dt
           ffac = np.sum(f_time, axis=1)/f0
           for i in range(0, len(f_time)):
               f_time[i] /= ffac[i]
           f_transport = f_transport - divF*dt
           return f_time, f_transport
          # return sol + dfdt(sol, t, u, cinvs)*dt

# Convert macroscopic solution from conserved variables to primitive variables
@njit
def ConToPri(sol, prisol):
	r = sol[:,0]
	ru = sol[:,1]
	E = sol[:,2]
	u = ru/r

	P = (gamma - 1.)*(E - 0.5*r*u**2)

	prisol[:,0] = r
	prisol[:,1] = u
	prisol[:,2] = P

	return prisol

# Convert macroscopic solution from primitive variables to conserved variables
@njit
def PriToCon(sol, consol):
	r = sol[:,0]
	u = sol[:,1]
	P = sol[:,2]

	ru = r*u
	E = P/(gamma - 1.) + 0.5*r*(u**2)

	consol[:,0] = r
	consol[:,1] = ru
	consol[:,2] = E

	return consol

# Setup operator matrices for flux reconstruction scheme
def createOperatorMatrices():
	global solGradMat, interpLeft_g, interpRight_g, solGradMat_g, leftcorrGrad_g, rightcorrGrad_g
	global leftcorrGrad, rightcorrGrad

	# Get local operator matrices
	interpLeft = generatemats.interpVect(basispts, -1.) # Interpolate solution to left face
	interpRight = generatemats.interpVect(basispts, 1.) # Interpolate solution to right face
	solGradMat = generatemats.solGradMat(basispts) # Evaluate gradient of solution at solution points
	leftcorrGrad = generatemats.corrGradVect(basispts, 'left') # Evaluate left correction function at solution points
	rightcorrGrad = generatemats.corrGradVect(basispts, 'right') # Evaluate right correction function at solution points

	# Create global operator matrices
	interpLeft_g   = np.zeros((nElems, npts))
	interpRight_g  = np.zeros((nElems, npts))
	solGradMat_g   = np.zeros((npts, npts))
	leftcorrGrad_g  = np.zeros((npts, nElems))
	rightcorrGrad_g = np.zeros((npts, nElems))

	stride = p+1
	for eidx in range(nElems):
		solGradMat_g[stride*eidx:stride*(eidx+1), stride*eidx:stride*(eidx+1)] = solGradMat
		interpLeft_g[eidx, stride*eidx:stride*(eidx+1)] = interpLeft
		interpRight_g[eidx, stride*eidx:stride*(eidx+1)] = interpRight
		if eidx != 0:
			leftcorrGrad_g[stride*eidx:stride*(eidx+1), eidx] = leftcorrGrad
		if eidx != nElems-1:
			rightcorrGrad_g[stride*eidx:stride*(eidx+1), eidx] = rightcorrGrad

	# Create divergence matrices
	# Left-biased operator
	DL = np.zeros((npts, npts))
	DL = solGradMat_g + leftcorrGrad_g @ (np.roll(interpRight_g, 1, axis=0) - interpLeft_g)
	DL /= detJ

	# Right-biased operator
	DR = np.zeros((npts, npts))
	DR = solGradMat_g + rightcorrGrad_g @ (np.roll(interpLeft_g, -1, axis=0) - interpRight_g)
	DR /= detJ

	return [DL, DR]

# Get spatial locations for nodal solution points in mesh
def getSpatialLocations(basispts, dom, nElems):
	ptsperelem = len(basispts)
	x = np.zeros(ptsperelem*nElems)
	for i in range(nElems):
		xi = dom[0] + i*(dom[1] - dom[0])/(nElems)
		xf = dom[0] + (i+1)*(dom[1] - dom[0])/(nElems)
		for j in range(ptsperelem):
			x[i*ptsperelem + j] = xi + (xf-xi)*0.5*(basispts[j]+1.)
	return x

#@njit
def run(f, t, dt, u, cinvs, tf):
    f_time = f
    f_transport = np.zeros_like(f)
    time, rho, rhoU, E = [], [], [], []
    while t < tf:
        ddt = min(dt, tf - t)
        [f_time, f_transport] = step(f_time, f_transport, t, ddt, u, cinvs)
        f_time, f_transport = limit(f_time), limit(f_transport)
        t += ddt
        
        time.append(t)
        rho.append(np.sum(f_time+f_transport))
        rhoU.append(np.sum((f_time+f_transport)@u))
        E.append(np.sum(0.5*(f_time+f_transport)@(u**2)))
        

    return f_time + f_transport, np.array(time), np.array(rho), np.array(rhoU), np.array(E)


print('Setting up simulation...')

# Setup mesh/solution points, initial conditions
[basispts, basiswts] = quadrules.getPtsAndWeights(p, 'gausslegendrelobatto')
x = getSpatialLocations(basispts, dom, nElems)
npts = nElems*(p+1)
detJ = (dom[1] - dom[0])/(2.0*nElems) # Element Jacobian
U = IC(x)

# Get upstream/downstream densities for post-processing
rl = U[0, 0]
rr = U[-1, 0]

# Setup phase space
f = np.ones((npts, Nx)) # (Spatial, velocity/internal energy)
global f0

ubounds = getVelocityBounds(U)
[u, phasewts] = setupVelocitySpace(Nx)

	
# Pre-compute collision invariants, generate operator matrices
cinvs = getCollisionInvariants(u)
[DL, DR] = createOperatorMatrices()

# Initialize velocity/internal energy distributions to equilibrium states
f = setEquilibriumDistribution(U, u, cinvs)
bc_fl = f[0,:]
bc_fr = f[-1,:]
f0 = np.sum(f, axis=1)

# Set reference states based on upstream states
rho_ref = U[0,0]
theta_ref = U[0,2]/U[0,0]
cs_ref = np.sqrt(gamma*theta_ref)
tau_ref = Kn*gamma/cs_ref/np.sqrt(gamma*np.pi/2.0)
# Compute time step from CFL condition and collision time
dt = (CFL/np.amax(ubounds))*(dom[1] - dom[0])/(nElems*(2*p+1))
dt = min(dt, tau_ref)

print(f'Time step 		    	: {dt}')
print(f'Ref. collision time 	: {tau_ref}')

# Run simulation
print('Running simulation...')
t = 0
[f, time, rho, rhoU, E] = run(f, t, dt, u, cinvs, tf)
plt.figure()
plt.plot(time, 100*((rho - rho[0])/rho[0]))
plt.plot(time, 100*((-rhoU[0]+rhoU)/rhoU[0]))
plt.plot(time, 100*((-E[0]+E)/E[0]))
plt.ylim(bottom=-16)
plt.xlabel('Time')
plt.ylabel('% error')
plt.legend(['Total Mass', 'Total Momentum', 'Total Energy'])
plt.title("Percent Error Versus Initial Value in Time, Integral Forcing, RK4")
plt.savefig("error_full_boltzmann_rk4.png")

# Plot solution
Ucon = np.zeros((npts, 3))
Ucon = getU(f, Ucon, u, cinvs)
U = np.zeros((npts, 3))
U = ConToPri(Ucon, U)

plt.figure()
plt.plot(x, U[:,0])
plt.plot(x, U[:,1])
plt.plot(x, U[:,-1])
#plt.ylim([0,4])
plt.xlabel('x')
plt.ylabel('Density/Velocity/Pressure')
plt.title("Density, Velocity, and Pressure in x at t = " + str(np.round(tf,2)) +", Integral Forcing, RK4")
plt.legend(['Density', 'Velocity', 'Pressure'])
#plt.savefig("rho_u_p_boltzmann_rk4_t1.png")


plt.figure()
grad_rho = 0.5*(DL + DR) @ U[:,0] # Compute using corrected gradients with centered averages (BR1 method)
plt.plot(x, grad_rho)
plt.xlabel('x')
plt.ylabel('Density gradient')
plt.title("Density Gradient in x at t = " + str(np.round(tf,2)) +", Integral Forcing, RK4")
#plt.savefig("densitygradient_boltzmann_rk4.png")

# Get inverse thickness ratio
lam = Kn # Assume reference length = 1
dd = (rr - rl)/np.max(grad_rho)
itr = lam/dd
print(f'Inverse thickness ratio : {itr}')
plt.show()

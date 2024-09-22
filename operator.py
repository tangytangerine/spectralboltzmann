import numpy as np
import scipy as sp
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt

def phi2(s, R):
	return 2*R*np.sinc(R*s)

n = 2**8+1
N = (n-1)//2
u = np.linspace(-1, 1, n)
# R = (u[-1] - u[1])*2/np.pi # Might be a different scaling factor here?
T = np.pi
R = 2*T/(1+3*np.sqrt(2))*2

f = np.zeros(n)
f[n//2 - n//10:n//2 + n//10] = 1
# f = np.sin(np.pi*(u+0.5)) + 1

plt.figure()
plt.plot(u, f)
freq = fftshift(fftfreq(n))
freq = freq*N/np.max(freq)

f0 = np.sum(f)

print(f'rho(t0) = {np.sum(f)}')
print(f'rhoU(t0) = {np.sum(f*u)}')
print(f'E(t0) = {np.sum(0.5*f*u**2)}')

rho_t0 = np.sum(f)
rhoU_t0 = np.sum(f*u)
E_t0 = np.sum(0.5*f*u**2)

dt = 1e-2
nt = 1000

errors_rho = []
errors_rhoU = []
errors_E = []

rho, rhoU, E = np.zeros(nt), np.zeros(nt), np.zeros(nt)

for ti in range(nt):
	rho[ti], rhoU[ti], E[ti] = np.sum(f), np.sum(f*u), np.sum(0.5*f*u**2)
    
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

	f += dt*Q

	ffac = np.sum(f)/f0
	f /= ffac # Hack to preserve density
    
	errors_rho.append((np.sum(f)-rho_t0)/rho_t0 * 100)
	errors_rhoU.append((np.sum(f*u) - rhoU_t0)/rhoU_t0 * 100)
	errors_E.append((np.sum(0.5*f*u**2) - E_t0)/E_t0 * 100)
    
plt.plot(u, f)

print(f'rho(t_f) = {np.sum(f)}')
print(f'rhoU(t_f) = {np.sum(f*u)}')
print(f'E(t_f) = {np.sum(0.5*f*u**2)}')
print()
print(f'Percent Error in E = {np.round(np.abs(E_t0 - np.sum(0.5*f*u**2))/E_t0 * 100, 3)}')

a = 0.4141
def shiftfunc(x):
    return 3/2*a*2*x - 2*2
shift = sp.optimize.fsolve(shiftfunc, 1)[0]/4.8
#maxwellian = f[n//2]-np.sqrt(2/np.pi)*(u+shift)**2/a**3*np.exp(-(u+shift)**2/2/a**2)
# maxwellian[0:47], maxwellian[205:] = 0, 0
maxwellian = np.sqrt(2/np.pi)*(2.55*u+shift)**2/a**3*np.exp(-(2.55*u+shift)**2/2/a**2)
maxwellian[:np.argmin(maxwellian[:n//2])] = 0
#maxim = np.max(maxwellian)
#maxwellian = maxim - maxwellian
#maxwellian[:np.argmin(maxwellian[:n//2])], maxwellian[n//2+np.argmin(maxwellian[n//2:]):] = 0,0
plt.plot(u, maxwellian, '-.')

plt.xlabel('u')
plt.ylabel('f')
plt.legend(['f (t=0)', 'f (t='+str(int(nt*dt))+')', 'Maxwellian'])
plt.title("Initial and Final Boltzmann Distributions at t="+str(int(nt*dt)))
plt.savefig("initial_final_integral.png")
plt.show()

plt.figure()
plt.plot(np.arange(0, nt)*dt, np.array(errors_rho))
plt.plot(np.arange(0, nt)*dt, np.array(errors_rhoU))
plt.plot(np.arange(0, nt)*dt, np.array(errors_E))
plt.xlabel('Time')
plt.ylabel('% error')
plt.legend(['Total Mass', 'Total Momentum', 'Total Energy'])
plt.title("Percent Error Versus Initial Value in Time")
plt.savefig("percenterror_integral.png")

plt.figure()
plt.semilogy(np.arange(0, nt)*dt, np.abs(rho))
plt.semilogy(np.arange(0, nt)*dt, np.abs(rhoU))
plt.semilogy(np.arange(0, nt)*dt, np.abs(E))
plt.legend(['rho', 'rho*U', 'E'])
plt.xlabel("Time")
plt.ylabel('Density/Momentum/Energy')
plt.title('Values of Density, Momentum, and Energy in Time')
plt.savefig("vals_in_time_integral.png")
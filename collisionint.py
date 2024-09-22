import numpy as np
import scipy as sp
from scipy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt

# Compute Boltzmann integral by DFT
def boltzmann_int(f):
    
    f_hat = fft(f)
    f_freq = fftfreq(len(f_hat))
    
    N = 1
    k = -N/2
    R = 2/(3+np.sqrt(2))*np.pi
    
    n_pts = len(f)
    
    Q_hat = np.zeros(len(f_hat),dtype=np.complex_)
    
    def phi(index):
        return 2*R*np.sinc(R*index/np.pi)
    
    i = 0
    while k < N/2:
        l = 0
        
        while l < len(f_freq):
            l_freq = f_freq[l]
            m = 0
            
            while m < len(f_freq):
                m_freq = f_freq[m]
                if l_freq + m_freq == k:
                    
                    beta = R*np.pi*(phi(l) + phi(-m))
                    Q_hat[i] += f_hat[l]*f_hat[m]*beta
                    
                m += 1
                
            l += 1
            
        i += 1
        k += 1/len(f_hat)
        
    return ifft(Q_hat)

# Simple time stepping by Forward-Euler
def euler(f, dt):
    return f + dt*boltzmann_int(f)

# Rectangular function
domain = np.linspace(-5, 5, 50)
f = np.zeros(50)
v0 = 1.25

# Set up rectangular function such that f at |v| less than v0 is set as 1/2v_0
i = 0
for x in domain:
    if np.abs(x) <= v0:
        f[i] = 1/2/v0
    i += 1

dt = 0.1
times = np.arange(0, 50, dt)

for t in times:
    f = euler(f, dt)

ax = plt.plot(domain, f)
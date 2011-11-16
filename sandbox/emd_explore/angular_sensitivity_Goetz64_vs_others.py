# XXX This code is not finished!!!

import math

import matplotlib.numerix as nx
import pylab

import scipy

def G_q(zeta,delta_rho_q):
    # From, e.g. Neumann, 2002:                     nx.exp( -2*zeta**2 / delta_rho_q**2 )

    # Note, Viollet & Franceschini 99:              nx.exp( -zeta**2 / (2*delta_rho_q**2) )
    
    # J. Spaethe & Chittka 03 cite Snyder 79 with:  nx.exp( -2.77*(zeta/delta_rho_q)**2 )
    
    # Burton & Laughlin (2003) cite Snyder 79 with: nx.exp( -4*math.log(2)*abs(zeta)**2 / delta_rho_q**2 )
    
    # Is this exactly the same as Goetz's 1964 result?
    return nx.exp( -2*zeta**2 / delta_rho_q**2 )

def G_Fs(Fs, delta_rho):
    # Goetz 64
    return nx.exp( -3.56* (delta_rho*Fs)**2 )

delta_rho_q_deg =1.3
z_deg = pylab.linspace(-2.0,2.0,1024)

delta_rho_q = delta_rho_q_deg/180.0*math.pi
z = z_deg/180.0*math.pi
PSF_other = G_q(z,delta_rho_q)

Fs_cpd = pylab.linspace(0.0, 1.2, 4096*4) # cycles/degree
Fs = Fs_cpd*180/math.pi # cycles/rad
MTF_Goetz = G_Fs(Fs, delta_rho_q )

PSF_Goetz = scipy.ifft( MTF_Goetz )
Fs_max = Fs[-1]
delta = 1.0/(2*Fs_max)
z2_rad = scipy.arange( len(PSF_Goetz))*delta
z2_deg = z2_rad*180/math.pi

z2_deg = z2_deg[:200]*2
PSF_Goetz = PSF_Goetz[:200]*math.pi

pylab.subplot(2,1,1)
pylab.plot(z_deg,PSF_other,'b-')
pylab.plot(z2_deg,PSF_Goetz,'r-')
pylab.xlabel('angle (deg)')
pylab.ylabel('relative intensity')
pylab.setp( pylab.gca(), 'xlim', [-2,2] )

pylab.subplot(2,1,2)
pylab.plot(Fs_cpd,MTF_Goetz,'r-')
pylab.xlabel('Fs (cycles/degree)')
pylab.ylabel('relative intensity')

pylab.show()

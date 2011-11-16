#!/usr/bin/env python
import Numeric
from scipy_utils import prony # my stuff
import scipy
import scipy.signal as signal

plot = True
if plot:
    import pylab
    
def log_normal(t,K=1.0,tp=0.015,sigma=0.32):
    # defaults values for Eristalis (Howard, et al. 1984, (fig 6., not table 1)
    # see also Tatler, O'Carroll, and Laughlin 2000 (fig. 2)
    return K*scipy.exp(-(scipy.log(t/tp)**2)/(2*sigma**2))

def exp_decay(t,tau=0.008):
    # see Lindemann, et al. 2005
    # can be well fit with prony(V,0,1)
    return -1.0/tau * scipy.exp(-t/tau)

def james_lmc(t,
              a1=-1.06,tau1=0.012,sigma1=0.197,
              a2=0.167,tau2=0.021,sigma2=0.345):
    # see Lindemann, et al. 2005
    return a1*scipy.exp(-(scipy.log(t/tau1))**2/(2*sigma1**2))+a2*scipy.exp(-(scipy.log(t/tau2))**2/(2*sigma2**2))

hz = 2000.0
timestep = 1.0/hz
t = scipy.arange(0,.625+timestep,timestep)

#V = log_normal(t)
V = exp_decay(t,.5)
if 0:
    V = -V/sum(V)
    V[0] = V[0]+1.0 # highpass

#V = james_lmc(t)

#b,a = prony(V,3,2)
#b,a = prony(V,2,1)
b,a = prony(V,1,1) # for highpass
#b,a = prony(V,0,1) # for lowpass
#b,a = prony(V,20,20)
#b,a = prony(V,5,4)

input = Numeric.zeros(t.shape,Numeric.Float64)
input[0] = 1 # impulse response

output=signal.lfilter(b,a,input)
print 'V[:100]',V[:100]
print 'output[:100]',output[:100]

print repr(tuple(b))
print repr(tuple(a))

print 'close?',Numeric.allclose(V,output)

if 1:
    if 1:
        # FIR filter is costly to compute...
        b2 = V/scipy.sum(V)
        a2 = [1.0]
    output2=signal.lfilter(b2,a2,input)
    nax = 3
    print repr(tuple(b2))
    print repr(tuple(a2))
else:
    nax = 2
    
if plot:
    ax1 = pylab.subplot(nax,1,1)
    ax1.plot(t,V,'o-')
    ax1.set_ylabel('impulse response (input)')
    ax = pylab.subplot(nax,1,2,sharex=ax1,sharey=ax1)
    ax.plot(t,output,'o-')
    ax.set_ylabel('impulse response (output)')
    if 1:
        ax = pylab.subplot(nax,1,3)#,sharex=ax1,sharey=ax1)
        ax.plot(t,output2,'o-')
        ax.set_ylabel('impulse response (output2)')
    ax.set_xlabel('time (sec)')
    pylab.show()

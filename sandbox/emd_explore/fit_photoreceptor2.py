#!/usr/bin/env python
import Numeric
import scipy
import scipy.signal as signal

import fsee.EMDSim as EMDSim
prony = EMDSim.prony

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

hz = 500.0
timestep = 1.0/hz
t = scipy.arange(0,.625+timestep,timestep)

input = Numeric.zeros(t.shape,Numeric.Float64)
input[0] = 1 # impulse response

if 0:
    # test Lindemann 2005 equation 3
    dt = 1.0/hz
    nx = scipy
    tau=0.5
    max_t = tau*10
    t = nx.arange(0,max_t+dt,dt)
    V = -1.0/tau * nx.exp(-t/tau)
    V[0] = V[0]+1.0
    print 'nx.sum(V)',nx.sum(V) # should be 0.0 if true highpass

def make_filt():
    # like FilterMaker.iir_james_lmc
    dt = 1.0/hz
    max_t = .2 # 200 msec
    t = scipy.arange(0,max_t+dt,dt)
    V = james_lmc(t)
    b,a = EMDSim.prony(V,14,10)
    return (b,a), V
(b,a), V=make_filt()
output=signal.lfilter(b,a,input)
#print 'output[:100]',output[:100]
testlen = min(len(output),len(V))
assert scipy.allclose(output[:testlen],V[:testlen])

fm = EMDSim.FilterMaker(hz)
b2,a2 = fm.fir_highpass1(0.5)
print 'c'


##print repr(tuple(b))
##print repr(tuple(a))

##print 'close?',Numeric.allclose(V,output)

if 1:
    if 0:
        # FIR filter is costly to compute...
        b2 = V/scipy.sum(V)
        a2 = [1.0]
    output2=signal.lfilter(b2,a2,input)
    nax = 2
##    print repr(tuple(b2))
##    print repr(tuple(a2))
else:
    nax = 2
    
if plot:
    if 0:
        ax1 = pylab.subplot(nax,1,1)
        ax1.plot(t,V,'o-')
        ax1.set_ylabel('impulse response (input)')
        ax = pylab.subplot(nax,1,2,sharex=ax1,sharey=ax1)
    else:
        ax = pylab.subplot(nax,1,2)

    ax.plot(t[:output.shape[0]],output,'o-')
    ax.set_ylabel('impulse response (output)')
    if 1:
        ax = pylab.subplot(nax,1,1,sharex=ax,sharey=ax)
        #ax.plot(t,output2,'o-')
        #ax.set_ylabel('impulse response (output2)')
        ax.plot(t[:V.shape[0]],V,'o-')
        ax.set_ylabel('impulse response (input)')
    ax.set_xlabel('time (sec)')
    pylab.show()

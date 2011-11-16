#!/usr/bin/env python
import numpy
from fsee.EMDSim import prony # my stuff
import scipy
import scipy.signal as signal

import pylab
    
def log_normal(t,K=1.0,tp=0.020,sigma=0.355):
    return K*numpy.exp(-(numpy.log(t/tp)**2)/(2*sigma**2))

hz = 500.0
timestep = 1.0/hz
t = numpy.arange(0,.625+timestep,timestep)

tp=0.020
sigma=0.355
V = log_normal(t,tp=tp,sigma=sigma)
b,a = prony(V,11,11)

input = numpy.zeros(t.shape,numpy.Float64)
input[0] = 1 # impulse response

output=signal.lfilter(b,a,input)
#print 'V[:100]',V[:100]
#print 'output[:100]',output[:100]

print 'b=',repr(tuple(b))
print 'a=',repr(tuple(a))

print 'close?',numpy.allclose(V,output)

nax = 2
    
ax1 = pylab.subplot(nax,1,1)
orig,=ax1.plot(t,V,'o-')
ax1.set_ylabel('V')
ax1.legend([orig],[r'$V(t)=\rm{exp}[-((\rm{log}{t/\tau_p})^2)/{2\sigma^2}], \tau_p=\rm{%s}, \sigma=\rm{%s} $'%(str(tp),str(sigma))])
ax = pylab.subplot(nax,1,2,sharex=ax1,sharey=ax1)
filt,=ax.plot(t,output,'o-')
ax.legend([filt],['IIR filter'])
ax.set_ylabel('V')
ax.set_xlabel('time (sec)')
pylab.setp(ax1,'xlim',[0,.1])
pylab.show()

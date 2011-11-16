from __future__ import division

import scipy
import scipy.integrate
import scipy.interpolate
import scipy.signal
import math
import fsee.EMDSim as EMDSim

hz=5000.0
dur = 1.0
dt = 1.0/hz
times = scipy.arange( dur*hz )/hz
Vin = scipy.where( times > 0.1, 1.0, 0.0)

#iVin = scipy.interpolate.interp1d( times, Vin, kind='spline' )
iVin = scipy.interpolate.interp1d( times, Vin,
                                   kind='linear',
                                   bounds_error=False,
                                   )
tau = 0.01 # 10 msec

def dVdt(t,y):
    Vin_now = iVin(t)
    Vout_now = y[0]
    Vdiff = (Vin_now-Vout_now)
    result = Vdiff/tau
    return [result]

r = scipy.integrate.ode(dVdt)
r.set_initial_value([0.0],t=0.0)
y=[]
to = []
while r.successful() and r.t<times[-2]:
    r.integrate(r.t+dt)
    y.append(r.y)
    to.append(r.t)
y=scipy.array(y)
print y.shape

Vout = y[:,0]

fm = EMDSim.FilterMaker(hz,unity_gain=True)
b,a=fm.iir_lowpass1(tau)
Vout_iir = scipy.signal.lfilter(b,a,Vin)

import pylab

pylab.plot(times,Vin,'r-+',
           to,Vout,'g-+',
           times,Vout_iir,'b-+')
pylab.show()

from __future__ import division

import math
import scipy
#from compartments import Compartment, Simulation, StepFunction
from fsee.ncompart import Compartment, Simulation, StepFunction

# test to find lowpass filtering properties of chain of cylinders

n_cyls = 27
cyl_Cm = 1100.0/30.0/n_cyls
cyl_g_leak = 1100.0/n_cyls

# more single-compartment like
#g_a = 50000.0  # output ~36.5 msec
#g_a = 30000.0 # output ~38 msec
g_a = 20000.0 # input tau ~21 msec, output ~41 msec (high amp: ~8.25 msec input, 29 msec output)
#g_a = 1000.0 # input tau ~15.5 msec, output very long (high amp: 2 msec input, 97 msec output)
#g_a = 300.0
# less single-compartment like

axon_compartments = []
for i in range(n_cyls):
    cpt = Compartment(lambda t: 0.0, lambda t:0.0)
    cpt.Cm = cyl_Cm
    cpt.g_leak = cyl_g_leak
    if len(axon_compartments):
        cpt.link(axon_compartments[-1],g_a)
    axon_compartments.append(cpt)

if 1:
    if 0:
        hz = 500.0
        dur = 1.0
        times = scipy.arange( dur*hz )/hz
        g_e = scipy.where( scipy.alltrue( (times > 0.1,
                                           times<0.6)), 800, 0)
        g_i = scipy.where( scipy.alltrue( (times > 0.5,
                                               times<0.8)), 800, 0)
        ig_e = scipy.interpolate.interp1d(times,g_e,
                                          kind='linear',
                                          bounds_error=False)
        ig_i = scipy.interpolate.interp1d(times,g_i,
                                          kind='linear',
                                          bounds_error=False)
    else:
        ig_e = StepFunction(start=0.1,stop=0.6,amp=800.0)
        ig_i = StepFunction(start=0.5,stop=0.8,amp=800.0)

    if 0:
        hz=500.0
        dur = 1.0
        t = scipy.arange(hz*dur)/hz
        e = [ig_e(t0) for t0 in t]
        import pylab
        pylab.plot(t,e,'+-')
        pylab.show()
    axon_compartments[0].func_g_e = ig_e
    axon_compartments[0].func_g_i = ig_i

    sim = Simulation(axon_compartments)

    dt=1.0/500.0
    r = scipy.integrate.ode(sim.dVdt)
    r.set_integrator( 'vode',
                      nsteps = 1000,
                      #min_step=dt*1e-5,
                      max_step = dt,
                      atol=1e-7,
                      rtol=1e-3,
                      )
    r.set_initial_value(sim.get_initial_value(),t=0.0)

    all_Vs=[]
    times=[]
    while r.successful() and r.t<0.99:
        r.integrate(   r.t+dt )
        all_Vs.append( r.y    )
        times.append(  r.t    )
        print r.t
    all_Vs = scipy.array( all_Vs )

    import pylab

    ax=pylab.subplot(3,1,1)
    stime_line,=pylab.plot(times,[ig_e(t) for t in times],'r-+')
    stimi_line,=pylab.plot(times,[-ig_i(t) for t in times],'b-+')
    
    pylab.subplot(3,1,2,sharex=ax)
    inj_line,=pylab.plot(times,all_Vs[:,0],'r-+')
    
    pylab.subplot(3,1,3,sharex=ax)
    out_line,=pylab.plot(times,all_Vs[:,-1],'g-+')
    
    pylab.legend([inj_line,out_line],['input compartment','output compartment'])
    
    pylab.show()

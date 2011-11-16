from __future__ import division

import scipy
import scipy.integrate
import scipy.interpolate
import scipy.signal
import math
import fsee.EMDSim as EMDSim

class StepFunction:
    def __init__(self,start=0.1,stop=0.6,amp=1.0,ws=1e-8):
        self.start=start
        self.stop=stop
        self.amp=amp
        self.ws=ws

        self.start_onset = self.start-(self.ws*math.pi/2)
        self.start_offset = self.start+(self.ws*math.pi/2)
        self.stop_onset = self.stop-(self.ws*math.pi/2)
        self.stop_offset = self.stop+(self.ws*math.pi/2)
        
    def __call__(self,t):
        if t<=self.start_onset or self.stop_offset<t:
            return 0
        if t<=self.start_offset:
            trel = (t-self.start)/self.ws
            amp = math.sin(trel)*0.5+0.5
            return self.amp*amp
        if self.stop_onset<t:
            trel = (t-self.stop)/self.ws
            amp = math.sin(trel)*-0.5+0.5
            return self.amp*amp
        return self.amp

class Compartment:
    def __init__(self,func_g_e,func_g_i):
        
        self.Vm_t0 = 0.0 # initial value
        
        self.func_g_e = func_g_e
        self.func_g_i = func_g_i
        
        self.Cm = 1100.0/30.0 # Lindemann et al. (2005) Appendix
        
        self.g_leak = 1100
        self.g_e = 0 # placeholder
        self.g_i = 0 # placeholder

        self.E_leak = 0
        self.E_e = 30
        self.E_i = -30
        self.input_neighbors = [] # list of tuples (Compartment, g_a)
        
    def update_for_time(self,t):
        self.g_e = self.func_g_e(t)
        self.g_i = self.func_g_i(t)

    def link(self,other,g_a):
        assert isinstance(other,Compartment)
        # each inputs to the other
        self.input_neighbors.append( (other, g_a) )
        other.input_neighbors.append( (self, g_a) )
        
class Simulation:
    def __init__(self,compartments):
        self.compartments = compartments
    def get_initial_value(self):
        return scipy.array([c.Vm_t0 for c in self.compartments])
    def dVdt(self,t,V):
        assert len(self.compartments)==len(V)
        dVdt = scipy.zeros( V.shape, scipy.Float )
        for i in range(len(self.compartments)):
            Vi = V[i]
            cpt = self.compartments[i]
            cpt.update_for_time(t)
            
            I_mem = (cpt.g_leak*(Vi-cpt.E_leak) +
                     cpt.g_e*(Vi-cpt.E_e) +
                     cpt.g_i*(Vi-cpt.E_i))

            I_neighbors = 0.0
            
            for neighbor,g_a in cpt.input_neighbors:
                ni = self.compartments.index(neighbor)
                I_neighbors += g_a*(Vi-V[ni])

            dVdt[i] = -(I_mem+I_neighbors)/cpt.Cm
        return dVdt

def test():
    def create_soma():        
        hz = 500.0
        dur = 1.0
        times = scipy.arange( dur*hz )/hz
        g_e = scipy.where( scipy.alltrue( (times > 0.1,
                                           times<0.6)), 800, 0)
        g_i = scipy.where( scipy.alltrue( (times > 0.5,
                                           times<0.6)), 800, 0)

        ig_e = scipy.interpolate.interp1d(times,g_e,
                                          kind='linear',
                                          bounds_error=False)
        ig_i = scipy.interpolate.interp1d(times,g_i,
                                          kind='linear',
                                          bounds_error=False)
        return Compartment(ig_e,ig_i)

    def create_dendrite():        
        hz = 500.0
        dur = 1.0
        times = scipy.arange( dur*hz )/hz
        g_e = scipy.where( times > 0.65, 800, 0)
        g_i = scipy.where( times > 0.8, 800, 0)

        ig_e = scipy.interpolate.interp1d(times,g_e,
                                          kind='linear',
                                          bounds_error=False)
        ig_i = scipy.interpolate.interp1d(times,g_i,
                                          kind='linear',
                                          bounds_error=False)
        return Compartment(ig_e,ig_i)

    soma=create_soma()
    dendrite=create_dendrite()
    soma.link(dendrite,300)
    sim = Simulation([soma,dendrite])

    dt=1.0/500.0
    r = scipy.integrate.ode(sim.dVdt)
    r.set_integrator( 'vode', max_step = dt )
    r.set_initial_value(sim.get_initial_value(),t=0.0)

    all_Vs=[]
    times=[]
    while r.successful() and r.t<0.99:
        r.integrate(   r.t+dt )
        all_Vs.append( r.y    )
        times.append(  r.t    )
    all_Vs = scipy.array( all_Vs )

    import pylab

    soma_line,=pylab.plot(times,all_Vs[:,0],'r-+')
    dendrite_line,=pylab.plot(times,all_Vs[:,1],'g-+')
    pylab.legend([soma_line,dendrite_line],['soma','dendrite'])
    pylab.show()

if __name__=='__main__':
    test()

from __future__ import division

import math, sys
import numpy
import scipy
import scipy.signal as signal
import scipy.optimize as optimize
from fsee.ncompart import Compartment, Simulation, StepFunction, ConstantFunction
from fsee.EMDSim import EMDSim
import pylab

###############################################
# Step 1. Traditional correlator model
###############################################

hz = 200.0
dt = 1/hz
emd_sim = EMDSim(hz=hz,
                 n_receptors=4,
                 earlyvis_ba=([1.0],[]), # no early vision temporal filtering
                 emd_edges=[(0,1),(2,3)],
                 #emd_edges=[(1,0),(3,2)],
                 )
sinA_TF = 5.0
sinA_contrast = 1.0

sinB_TF = 5.0
sinB_contrast = 0.3

duration = 1.0 # seconds
time_now = 0.0 # seconds

trad_results = []
while time_now < duration:
    
    stimA1 = sinA_contrast*math.sin( time_now*2*math.pi*sinA_TF )
    stimA2 = sinA_contrast*math.sin( time_now*2*math.pi*sinA_TF + math.pi/2 )

    stimB1 = sinB_contrast*math.sin( time_now*2*math.pi*sinB_TF )
    stimB2 = sinB_contrast*math.sin( time_now*2*math.pi*sinB_TF + math.pi/2 )

    photoreceptors = [stimA1, stimA2, stimB1, stimB2]
    emds = emd_sim.step(photoreceptors)

    trad_results.append( [time_now]+list(emds)+
                         list(emd_sim.get_values('subunit_A_Bd'))+
                         list(emd_sim.get_values('subunit_Ad_B'))+
                         photoreceptors+
                         list(emd_sim.get_values('undelayed'))+
                         list(emd_sim.get_values('delayed'))
                         )
    time_now += dt

# name results

trad_results = numpy.asarray(trad_results) # make array
times = trad_results[:,0]
emdB = trad_results[:,2]
emdA = trad_results[:,1]
emdAa_bd = trad_results[:,3]
emdBa_bd = trad_results[:,4]
emdAad_b = trad_results[:,5]
emdBad_b = trad_results[:,6]
stimA1 = trad_results[:,7]
stimA2 = trad_results[:,8]
stimB1 = trad_results[:,9]
stimB2 = trad_results[:,10]

stimA1u = trad_results[:,11]
stimA2u = trad_results[:,12]
stimB1u = trad_results[:,13]
stimB2u = trad_results[:,14]

stimA1d = trad_results[:,15]
stimA2d = trad_results[:,16]
stimB1d = trad_results[:,17]
stimB2d = trad_results[:,18]

if 0:
    vals = [[
##        emdA,
##             stimA1,stimA2,
##             stimA1u,stimA2u,
##             stimA1d,stimA2d,
##             emdAa_bd,emdAad_b,
#             stimA1u,stimA2d,
#             stimA1u*stimA2d,
             (emdAa_bd,'Ae'),
             (emdAad_b,'Ai'),
             ],
            [
##        emdB,
##             stimB1,stimB2,
##             stimB1u,stimB2u,
##             stimB1d,stimB2d,
##             emdBa_bd,emdBad_b,
##             stimB1u,stimB2d,
##             emdBa_bd,
##             stimB1u*stimB2d,
             (emdBa_bd,'Be'),
             (emdBad_b,'Bi'),
             ]]
    ax = None
    for i,v1 in enumerate(vals):
        ax = pylab.subplot(3,1,i+1,sharex=ax,sharey=ax)
        for pv,name in v1:
            pylab.plot(times,pv,label=name)
        pylab.legend()    
            
###############################################
# Step 1.5. Create spline samplers
###############################################

class CSplineInterp:
    def __init__(self,x,y,gain=1.0):
        self.dx=x[1]-x[0]
        self.x0=x[0]
        xtest = numpy.arange(len(y))*self.dx + self.x0
        if not numpy.allclose(xtest,x):
            raise ValueError('x must be equally spaced')
        self.y=y
        self.cj = signal.cspline1d(y)
        self.gain=gain
    def __call__(self,newx):
        nx = numpy.array([newx],dtype=numpy.Float)
        nv = signal.cspline1d_eval(self.cj,nx,dx=self.dx,x0=self.x0)
        return self.gain*nv[0]

if 0:
    interp = CSplineInterp(times,emdAa_bd)
    newtimes = numpy.arange(0.0,1.0,0.001)
    emdAa_bd_interp = numpy.array([interp(nt) for nt in newtimes])
    pylab.plot(times,emdAa_bd,'b+')
    pylab.plot(newtimes,emdAa_bd_interp,'r-')
    pylab.show()
    sys.exit(0)

def rectify(sig):
    cond1 = sig < 0
    cond2 = ~cond1
    rsig = numpy.zeros_like(sig)
    rsig[cond1] = 0.0
    rsig[cond2] = sig[cond2]
    return rsig

if 0:
    emdAe_func = CSplineInterp(times,emdAa_bd)
    emdBe_func = CSplineInterp(times,emdBa_bd)
    emdAi_func = CSplineInterp(times,emdAad_b) # flip sign because driving potential is negative
    emdBi_func = CSplineInterp(times,emdBad_b)
else:
    emdAe_func = CSplineInterp(times,rectify(emdAa_bd))
    emdBe_func = CSplineInterp(times,rectify(emdBa_bd))
    emdAi_func = CSplineInterp(times,rectify(emdAad_b)) # flip sign because driving potential is negative
    emdBi_func = CSplineInterp(times,rectify(emdBad_b))

    if 1:
        nx = numpy
        print 'cpt mean A ge',nx.mean(nx.array([emdAe_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print 'cpt mean A gi',nx.mean(nx.array([emdAi_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print 'cpt mean B ge',nx.mean(nx.array([emdBe_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print 'cpt mean B gi',nx.mean(nx.array([emdBi_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print
        print 'cpt std A ge',nx.std(nx.array([emdAe_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print 'cpt std A gi',nx.std(nx.array([emdAi_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print 'cpt std B ge',nx.std(nx.array([emdBe_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print 'cpt std B gi',nx.std(nx.array([emdBi_func(t) for t in nx.arange(0.0,5.0,0.005)]))
        print

if 1:
    # prove that resampling works
    newtimes = numpy.r_[0.0:1.0:0.001] # 1000 hz
    def supersample(interp):
        return numpy.array([interp(t) for t in newtimes])
    ax=pylab.subplot(3,1,1)
    Ae = supersample(emdAe_func)
    Ai = supersample(emdAi_func)# sign flipped to show negative driving potential
    pylab.plot( newtimes, Ae, label='Ae' )
    pylab.plot( newtimes, Ai, label='Ai' ) 
    pylab.plot( newtimes, Ae-Ai, label='Ae-Ai' )
    pylab.legend()
    ax2=pylab.subplot(3,1,2,sharex=ax,sharey=ax)
    Be = supersample(emdBe_func)
    Bi = supersample(emdBi_func)# sign flipped to show negative driving potential
    pylab.plot( newtimes, Be, label='Be' )
    pylab.plot( newtimes, Bi, label='Bi' ) 
    pylab.plot( newtimes, Be-Bi, label='Be-Bi' )
    pylab.legend()
    pylab.draw()

###############################################
# Step 2. Play subunit activities into compartments
###############################################

def objective_func(params):
    #g_a, gscale, CmAB, CmX, g_leakAB, g_leakX = params
    g_a, gscale, membrane_area_AB, membrane_area_X = params
    #g_a, gscale = params
    #CmAB, CmX, g_leakAB, g_leakX = 1100.0/90, 1100.0/90, 1100.0/30, 1100.0/30
    CmAB, g_leakAB = membrane_area_AB*12.2, membrane_area_AB*33.0
    CmX, g_leakX = membrane_area_X*12.2, membrane_area_X*33.0
    g_a = abs(g_a)
    gscale = abs(gscale)
    CmAB = abs(CmAB)
    CmX = abs(CmX)
    g_leakAB = abs(g_leakAB)
    g_leakX = abs(g_leakX)
    #print 'g_a, gscale, g_leakAB, g_leakX, CmAB, CmX',g_a, gscale, g_leakAB, g_leakX, CmAB, CmX
    print 'g_a, gscale, membrane_area_AB, membrane_area_X',g_a, gscale, membrane_area_AB, membrane_area_X
    
    n_abx = 3 # test
    n_chain = 0 # hanging off cptX to get more "weight"
    n_cyls = n_abx+n_chain
    
    #cyl_Cm = 1100.0/30.0/n_cyls
    #cyl_g_leak = 1100.0/n_cyls
    #cyl_Cm = Cm/n_cyls

    # create "cell body"
    cptX = Compartment(ConstantFunction(0.0),
                       ConstantFunction(0.0),
                       name='cpt X')
    cptX.Cm = CmX
    cptX.g_leak = g_leakX

    # create dendritic compartments

    # (for now ge and gi are from steady-state values of correlator subunits calculated below)
    emdAe_func.gain=gscale
    emdAi_func.gain=gscale
    cptA = Compartment(emdAe_func,
                       emdAi_func,
                       name='cpt A')
    cptA.Cm = CmAB
    cptA.g_leak = g_leakAB
    cptA.link( cptX, g_a )

    emdBe_func.gain=gscale
    emdBi_func.gain=gscale
    cptB = Compartment(emdBe_func,
                       emdBi_func,
                       name='cpt B')
    cptB.Cm = CmAB
    cptB.g_leak = g_leakAB
    cptB.link( cptX, g_a )

    cpts=[cptX,cptA,cptB]
    assert n_abx == len(cpts)

    for i in range(n_chain):
        # hanging off cptX to give it more "weight"
        cpt = Compartment(ConstantFunction(0.0),
                          ConstantFunction(0.0),
                          name='extra%d'%(i+1))
        cpt.Cm = CmX
        cpt.g_leak = g_leakX
        cpt.link( cptX, g_a )

        cpts.append(cpt)

    sim = Simulation(cpts)
    print sim
    dt=1.0/hz
    r = scipy.integrate.ode(sim.dVdt)
    r.set_integrator( 'vode',
                      nsteps = 10000,
                      #min_step=dt*1e-5,
                      max_step = dt,
                      atol=1e-7,
                      rtol=1e-3,
                      )
    r.set_initial_value(sim.get_initial_value(),t=0.0) 
    all_Vs=[]
    cpt_times=[]
    while r.successful() and r.t<(duration-dt):
        r.integrate(   r.t+dt )
        all_Vs.append( r.y    )
        cpt_times.append(  r.t    )

    all_Vs = numpy.array( all_Vs )
    
    VX = all_Vs[:,0]
    VA = all_Vs[:,1]
    VB = all_Vs[:,2]

    if 0:
        distXA = numpy.mean((VX-VA)**2)
        distXB = numpy.mean((VX-VB)**2)
        result = distXA-distXB + 1.0/numpy.mean(VX)
    elif 0:
        VX = numpy.mean(VX)
        VA = numpy.mean(VA)
        VB = numpy.mean(VB)
        
        distXA = abs(VX-VA)
        distXB = abs(VX-VB)
        result = distXA-distXB# + abs(1.0/gscale)
    elif 1: # good!
        cond1 = cpt_times>0.1

        VX = numpy.mean(VX[cond1])
        VA = numpy.mean(VA[cond1])
        VB = numpy.mean(VB[cond1])
        
        distXA = abs(VX-VA)/(VX+VA)
        distXB = (VB-VX)/(VX+VB)
        result = (2*distXA) + distXB + abs(1.0/g_a)
        #result = distXA-distXB# + abs(1.0/gscale)
    elif 0:
        VX = numpy.mean(VX)
        VA = numpy.mean(VA)
        VB = numpy.mean(VB)

        result = VA/VX + VB/VX #(VB/VX)-(VA/VX)
        
        
        distXA = abs(VX-VA)
        distXB = abs(VX-VB)
        
    elif 0:
        # only works when VB < VX < VA
        VX = numpy.mean(VX)
        VA = numpy.mean(VA)
        VB = numpy.mean(VB)

        result = (1.0 - (VA-VX)/VA) + ((VX-VB)/VX) # maximize fraction of VA that VX is and minimize fraction of VX that VB is
        
        distXA = abs(VX-VA)
        distXB = abs(VX-VB)
        
    print 'distXA',distXA
    print 'distXB',distXB
    print 'distXA-distXB',distXA-distXB
    print 'result',result
    print
    if 1:
        pylab.ion()
        pylab.subplot(3,1,3,sharex=ax)
        pylab.cla()
        for i in range(n_abx):
            pylab.plot(cpt_times,all_Vs[:,i],label=cpts[i].name)
        pylab.legend()
        pylab.title(str((g_a, gscale, g_leakAB, g_leakX, CmAB, CmX)))
        pylab.draw()
        
    return result

###g_a, gscale = 200.0, 1100.0
###CmAB, CmX, g_leakAB, g_leakX = 30.0, 30.0, 1100.0, 300.0
##g_a, gscale, g_leakAB, g_leakX, CmAB, CmX = 105.013825939, 1598.29454578, 2121.47363238, 0.221191126571, 9.55716147243, 14.8416114134
##membrane_area_AB = 1.0
##membrane_area_X = 1.0
g_a, gscale, membrane_area_AB, membrane_area_X = 168.517001355, 1624.3165492, 1.33801975775, 0.00170394152798
params = g_a, gscale, membrane_area_AB, membrane_area_X
#params = g_a, gscale, CmAB, CmX, g_leakAB, g_leakX
#objective_func(params)
if 0:
    final = optimize.fmin( objective_func, params )
    print 'repr(final)',repr(final)
else:
    objective_func( params )
pylab.show()

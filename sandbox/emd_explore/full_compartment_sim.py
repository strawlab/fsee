from __future__ import division

import math, os, glob, sys
import scipy
scipy.pkgload()
import numarray
import numpy as nx
import scipy.signal as signal
import tables
from fsee.ncompart import Compartment, Simulation, StepFunction
import fsee.eye_geometry.cyl_proj as cyl_proj

class CSplineInterp:
    def __init__(self,x,y,gain=1.0):
        self.dx=x[1]-x[0]
        self.x0=x[0]
        xtest = nx.arange(len(y))*self.dx + self.x0
        if not nx.allclose(xtest,x):
            raise ValueError('x must be equally spaced')
        self.y=y
        self.cj = signal.cspline1d(y)
        self.gain=gain
    def __call__(self,newx):
        newx = nx.array([newx],dtype=nx.Float)
        nv = signal.cspline1d_eval(self.cj,newx,dx=self.dx,x0=self.x0)
        return self.gain*nv[0]

#####################################
# find spatial distribution of EMD inputs to each compartment
#####################################

def get_EMD_compartments_info():
    # remove 1 horizontally to make line up with EMDs
    n_per_side = 15
    n_compartments_vert = int(math.ceil(len(cyl_proj.theta)/n_per_side))
    n_compartments_horiz = int(math.ceil((len(cyl_proj.phi)-1)/n_per_side))

    all_vert_weights=[]
    for i in range(n_compartments_vert):
        i0 = i*n_per_side
        i1 = (i+1)*n_per_side
        if i1>=len(cyl_proj.theta):
            i1 = len(cyl_proj.theta)-1
        vert_weights = nx.zeros( (len(cyl_proj.theta),), nx.UInt8 )
        vert_weights[i0:i1]=1
        all_vert_weights.append( vert_weights )

    all_horiz_weights=[]
    for j in range(n_compartments_horiz):
        j0 = j*n_per_side
        j1 = (j+1)*n_per_side
        if j1>=(len(cyl_proj.phi)-1):
            j1 = len(cyl_proj.phi)-2
        horiz_weights = nx.zeros( (len(cyl_proj.phi)-1,), nx.UInt8 )
        horiz_weights[j0:j1]=1
        all_horiz_weights.append( horiz_weights )

    all_compartment_inputs = []
    for vert_weights in all_vert_weights:
        for horiz_weights in all_horiz_weights:
            compartment_inputs = nx.outerproduct( vert_weights, horiz_weights )
            compartment_inputs = nx.ravel( compartment_inputs )
            all_compartment_inputs.append( compartment_inputs )

    n_EMD_cyls = len(all_compartment_inputs)
    print '%d EMD compartments (%dx%d), each %dx%d EMDs'%(
        n_EMD_cyls,
        len(all_vert_weights),
        len(all_horiz_weights),
        n_per_side,n_per_side)
    return n_EMD_cyls, all_compartment_inputs

def setup_compartments(n_EMD_cyls, all_compartment_inputs,
                       g_a, gscale,
                       membrane_area_AB,
                       membrane_area_X,
                       image_filename, vel_dps):
    
    ###########################
    # load EMD modeling results
    ###########################

    h5fname = 'sim_results_%s_%f.h5'%(image_filename,vel_dps)
    SAVEDIR = '/home/astraw/FULL_COMPARTMENT_SIM'
    #SAVEDIR = '/home/astraw/SMALL_COMPARTMENT_SIM'
    print 'opening',h5fname
    h5file = tables.openFile(os.path.join(SAVEDIR,h5fname),mode='r+')

    weights_A = h5file.root.info.cols.weights_A[0]
    weights_B = h5file.root.info.cols.weights_B[0]

    table = h5file.root.conductance
    print 'reading data...'
    sys.stdout.flush()
    times = table.cols.cur_time
    subunit_A_Bd = table.cols.directionA
    subunit_Ad_B = table.cols.directionB

    duration_sec = times[-1]
    
    # for each timestep:
    #   rectify each EMD input
    #   scale according to spatial weight map (previously done)
    #   sum conductance from each EMD onto compartment (do here)

    print 'rectifying...'
    sys.stdout.flush()

    # rectify
    re = nx.where( subunit_A_Bd > 0, subunit_A_Bd , 0 )
    ri = nx.where( subunit_Ad_B > 0, subunit_Ad_B , 0 )

    print 'scaling'
    sys.stdout.flush()

    # scale
    sre = weights_A*re
    sri = weights_B*ri

    print 'sre.shape',sre.shape
    print 'type(sre)',type(sre)
    print 'weights_A.shape',weights_A.shape
    print 're.shape',re.shape

    # sum conductances onto compartment
    cpt_interps = []
    for compartment_inputs in all_compartment_inputs:
        g_e_compartment = nx.sum(sre*compartment_inputs,axis=1)
        g_i_compartment = nx.sum(sri*compartment_inputs,axis=1)
        interp_g_e = CSplineInterp( times, g_e_compartment, gain=gscale/400000.0 )
        interp_g_i = CSplineInterp( times, g_i_compartment, gain=gscale/400000.0 )
        if 1:
            print 'cpt mean ge',nx.mean(nx.array([interp_g_e(t) for t in nx.arange(0.0,duration_sec,0.005)]))
            print 'cpt mean gi',nx.mean(nx.array([interp_g_i(t) for t in nx.arange(0.0,duration_sec,0.005)]))
            print 'cpt std ge',nx.std(nx.array([interp_g_e(t) for t in nx.arange(0.0,duration_sec,0.005)]))
            print 'cpt std gi',nx.std(nx.array([interp_g_i(t) for t in nx.arange(0.0,duration_sec,0.005)]))
            print
        cpt_interps.append( (interp_g_e,interp_g_i))

    ############################################
    # setup compartments
    ############################################

    n_axon_cyls = 1
    n_cyls = n_EMD_cyls + n_axon_cyls
    cyl_Cm = 1100.0/30.0/n_cyls # membrane capacitance (total = 1100/30)
    cyl_g_leak = 1100.0/n_cyls  # membrane conductance (total = 1100)

    axon_cyls = []
    for i in range(n_axon_cyls):
        CmX, g_leakX = membrane_area_X*12.2, membrane_area_X*33.0
        cpt = Compartment(lambda t: 0.0, lambda t:0.0)
        cpt.Cm = CmX
        cpt.g_leak = g_leakX
        if len(axon_cyls):
            cpt.link(axon_cyls[-1],g_a)
        axon_cyls.append(cpt)

    EMD_cyls = []
    for i in range(n_EMD_cyls):
        CmAB, g_leakAB = membrane_area_AB*12.2, membrane_area_AB*33.0
        interp_g_e,interp_g_i = cpt_interps[i] # conducances stored above
        cpt = Compartment(lambda t: 0.0, lambda t:0.0)
        cpt.func_g_e = interp_g_e
        cpt.func_g_i = interp_g_i
        cpt.Cm = CmAB
        cpt.g_leak = g_leakAB
        cpt.link(axon_cyls[0],g_a) # link to first axon compartment
        EMD_cyls.append(cpt)

    all_cyls = EMD_cyls + axon_cyls
    
    return h5file, all_cyls, duration_sec

def doit(image_filename,vel_dps):
    g_a, gscale, membrane_area_AB, membrane_area_X = 168.517001355, 1624.3165492, 1.33801975775, 0.00170394152798
    
    n_EMD_cyls, all_compartment_inputs = get_EMD_compartments_info()
    print 'n_EMD_cyls',n_EMD_cyls
    h5file, all_cyls, duration_sec = setup_compartments(n_EMD_cyls,
                                                        all_compartment_inputs,
                                                        g_a, gscale,
                                                        membrane_area_AB,
                                                        membrane_area_X,
                                                        image_filename, vel_dps)
                                          
    sim = Simulation(all_cyls)
    print sim
    dt=1.0/500.0
    print 'dt',dt
    r = scipy.integrate.ode(sim.dVdt)
    r.set_integrator( 'vode',
                      nsteps = 100000,
                      #min_step=dt*1e-5,
                      max_step = dt,
                      atol=1e-6,
                      rtol=1e-3,
                      )
    r.set_initial_value(sim.get_initial_value(),t=0.0)
    
    all_Vs=[]
    times=[]
    try:
        while r.successful() and r.t<duration_sec-dt:
            r.integrate(   r.t+dt )
            all_Vs.append( r.y    )
            times.append(  r.t    )
            print r.t
    except Exception,x:
        print 'exception in simulation',str(x)
        failed = True
    else:
        failed = False
        
    all_Vs = numarray.array( all_Vs )
    times = numarray.array( times )

    # save results
    if hasattr(h5file.root,'sim_results'):
        h5file.removeNode(h5file.root,'sim_results',recursive=True)
    sim_results=h5file.createGroup(h5file.root, "sim_results", "results of compartmental modeling" )
    h5file.createArray(sim_results, "n_input_compartments", [n_EMD_cyls])
    print 'all_Vs',all_Vs
    h5file.createArray(sim_results, "all_Vs", all_Vs)
    h5file.createArray(sim_results, "times", times)
    h5file.close()
    
def main():
    if len(sys.argv)==3:
        image_filename = sys.argv[1]
        vel_dps = float(sys.argv[2])
    else:
        1/0
        # more single-compartment like
        #g_a = 50000.0  # output ~36.5 msec
        #g_a = 30000.0 # output ~38 msec
        g_a = 20000.0 # input tau ~21 msec, output ~41 msec (high amp: ~8.25 msec input, 29 msec output)
        #g_a = 1000.0 # input tau ~15.5 msec, output very long (high amp: 2 msec input, 97 msec output)
        #g_a = 300.0
        # less single-compartment like

        #results_files = glob.glob('/mnt/backup/andrew/lindemann/*.h5')
        #results_file = results_files[0]
        image_filename = '080499-2048x316.bmp'
        vel_dps = -151.269215
    print 'simulating:',image_filename,vel_dps
    doit(image_filename,vel_dps)

if __name__=='__main__':
    main()

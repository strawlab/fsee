import glob, os, math
import Image
import numpy as nx
import numpy as np
import scipy.signal as signal
import scipy.ndimage as nd_image
import pickle
import weight_map

SAVE_HDF = True
if SAVE_HDF:
    import tables

    SAVEDIR = os.path.expanduser('~/FULL_COMPARTMENT_SIM')
    #SAVEDIR = '/home/astraw/SMALL_COMPARTMENT_SIM'
    
import fsee.eye_geometry.cyl_proj as cyl_proj
import fsee.EMDSim as EMDSim

R2D = 180.0/math.pi

DBG=0
if DBG:
    import pylab
    pylab.ion()

class Im2R:
    def __init__(self,active_imshape,dtype):
        self.dtype = dtype
        self.im = np.zeros( cyl_proj.imshape, self.dtype )
        height_diff = cyl_proj.imshape[0]-active_imshape[0]
        self.start_i = height_diff//2
        self.end_i = self.start_i+active_imshape[0]
        self.extent = np.array((cyl_proj.imphi[0], cyl_proj.imphi[-1],
                                cyl_proj.imtheta[self.start_i], cyl_proj.imtheta[self.end_i-1]))*R2D
    def __call__(self,image):
        self.im[self.start_i:self.end_i,:]=image.astype(self.dtype)
        im = np.ravel(self.im)
        resps = cyl_proj.im2receptors*im
        resps.shape = len(theta),len(phi)
        return resps

class Im2RFastButBroken:
    def __init__(self,active_imshape,dtype):
        self.dtype = dtype
        height_diff = cyl_proj.imshape[0]-active_imshape[0]
        self.start_i = height_diff//2
        self.end_i = self.start_i+active_imshape[0]
        # slicing doesn't work (yet)
        self.im2receptors = cyl_proj.im2receptors[self.start_i:self.end_i,:]
        self.extent = np.array((cyl_proj.imphi[0], cyl_proj.imphi[-1],
                                cyl_proj.imtheta[self.start_i], cyl_proj.imtheta[self.end_i-1]))*R2D
    def __call__(self,image):
        imflat = np.ravel(image).astype(self.dtype)
        resps = self.im2receptors*imflat
        resps.shape = len(theta),len(phi)
        return resps
    
phi = cyl_proj.phi
theta = cyl_proj.theta
eye = cyl_proj.eye

def setup_emd_sim(hz,fname,vel_dps,n_tsteps):

    ################################
    # setup filters
    ################################

    fm = EMDSim.FilterMaker(hz)#,unity_gain=True)
    emd_lp_ba = fm.iir_lowpass1(tau=0.010) # Lindemann (2005) p. 6438
    emd_hp_ba = fm.iir_highpass1(tau=0.060)
    earlyvis_ba = fm.james_lmc(a1=-1.06,tau1=0.012,sigma1=0.197,
                               a2=0.167,tau2=0.021,sigma2=0.345)
    n_receptors = cyl_proj.im2receptors.shape[0]

    ################################
    # setup EMD baselines
    ################################

    emd_edges = []

    for i,theta_i in enumerate(theta):
        for j,phi_j in enumerate(phi):
            if j==0:
                continue
            R_idx = i*len(phi)+j   # index into receptors
            R_idx_prev = i*len(phi)+(j-1) # index into receptors
            #emd_edges.append( (R_idx,R_idx_prev) )
            emd_edges.append( (R_idx_prev,R_idx) )
    emds_shape = len(theta), len(phi)-1
    
    ################################
    # setup spatial weight map
    ################################
    
    weightmap = weight_map.make_weight_map(eye)
    
    ################################
    # create motion detection model
    ################################

    print 'using %d EMDs'%len(emd_edges)
    emd_sim = EMDSim.EMDSim(hz=hz,
                            earlyvis_ba = earlyvis_ba,
                            n_receptors = n_receptors,
                            emd_edges = emd_edges,
                            emd_lp_ba = emd_lp_ba,
                            emd_hp_ba = emd_hp_ba,
                            #subtraction_imbalance = 1.0,
                            lindemann_weight_map = weightmap,
                            )

    weights_A = emd_sim.get_values('weights_A')
    weights_B = emd_sim.get_values('weights_B')
    
    membrane_lp_ba = fm.iir_lowpass1(tau=0.008)
    
    compartment = EMDSim.SingleCompartmentSim(weights_A,weights_B,membrane_lp_ba)

    #################################
    # setup HDF5 saver using pytables
    #################################

    arrshape = weights_A.shape
    # pytables row description
    ConductanceTimeStep = {
        'cur_time':tables.FloatCol(pos=0),
        'directionA':tables.Float32Col(shape=arrshape,pos=1),
        'directionB':tables.Float32Col(shape=arrshape,pos=2),
        'EMD_outputs':tables.Float32Col(shape=arrshape,pos=3),
        'Vm':tables.Float32Col(pos=4),
        }
    Info = {
        'fname'  : tables.StringCol(80,pos=0),
        'vel_dps': tables.Float32Col(pos=1),
        'weights_A':tables.Float32Col(shape=arrshape,pos=2),
        'weights_B':tables.Float32Col(shape=arrshape,pos=3),
        }

    save_fname = 'sim_results_%s_%f.h5'%(fname,vel_dps)
    print 'saving to',save_fname
    save_fname = os.path.join(SAVEDIR,save_fname)
    try:
        h5file = tables.openFile(save_fname,mode='r+',title='conductance information')
    except IOError,x:
        print 'creating',save_fname
        h5file = tables.openFile(save_fname,mode='w',title='conductance information')
    if hasattr(h5file.root,'info'):
        h5file.removeNode(h5file.root,'info')
    info = h5file.createTable( h5file.root, 'info', Info,
                               'simulation information',
                               expectedrows=1 )
    row = info.row
    row['fname']=fname
    row['vel_dps']=vel_dps
    row['weights_A'] = weights_A
    row['weights_B'] = weights_B
    row.append()
    info.flush()
    if hasattr(h5file.root,'conductance'):
        h5file.removeNode(h5file.root,'conductance')
    saver = h5file.createTable( h5file.root, 'conductance', ConductanceTimeStep,
                                'conductance time steps',
                                expectedrows=n_tsteps )
    #################################

    return emd_sim, compartment, emds_shape, saver, h5file

def doit(args):
    imfile, vel_dps = args
    print 'vel_dps',vel_dps
    vel_pps = vel_dps/360.0*2048.0
    hz = 500.0
    vel_pix_per_dt = vel_pps/hz
    print 'vel_pix_per_dt',vel_pix_per_dt

    duration_sec = 5.0
    
    im = Image.open(imfile)
    fname = os.path.split(imfile)[-1]

    n_tsteps = duration_sec*hz+1
    emd_sim, compartment, emds_shape, saver, h5file = setup_emd_sim(hz,fname,vel_dps,n_tsteps)

    weights_A = emd_sim.get_values('weights_A')
    weights_B = emd_sim.get_values('weights_B')
    
    if 1:
        imnx = np.fromstring(im.tostring('raw','RGB',0,-1),np.uint8)
    else:
        # avoid MemoryError in scipy at the moment
        #imnx = np.fromstring(im.tostring('raw','RGB',0,-1),np.uint8)
        imnx = np.fromstring(im.tostring('raw','RGB',0,-1),np.uint8)
        imnx = np.asarray(imnx)
    imnx.shape = im.size[1], im.size[0], 3
    print fname, imnx.shape

    dtype = np.float
    #dtype = np.uint8

    i2 = imnx[:,:,1]
    im2r = Im2R(i2.shape,dtype)
    shifted = np.ones( i2.shape, dtype ) # uses nd_image
    emds = np.zeros( emds_shape, np.float )
    ni = im2r(shifted)

    if DBG:
        pylab.figure()
        pylab.subplot(7,1,1)
        pylab.title('orig')
        axim_shifted = pylab.imshow( shifted, vmin=0.0, vmax=255.0,
                                     extent = im2r.extent,
                                     )
        pylab.colorbar()

        pylab.subplot(7,1,2)
        pylab.title('retina')
        axim_ni = pylab.imshow( ni, vmin=0.0, vmax=255.0,
                                extent=np.array((phi[0],phi[-1],theta[0],theta[-1]))*R2D )
        pylab.colorbar()

        pylab.subplot(7,1,3)
        pylab.title('earlyvis')
        axim_earlyvis = pylab.imshow( ni, vmin=-255.0, vmax=255.0,
                                      extent=np.array((phi[0],phi[-1],theta[0],theta[-1]))*R2D )
        pylab.colorbar()

##        vmin=-(255.0**2)
##        vmax=255.0**2
        vmin=-255.0
        vmax=255.0
        pylab.subplot(7,1,4)
        pylab.title('delayed')
        axim_delayed = pylab.imshow( ni, vmin=vmin, vmax=vmax,
                                     extent=np.array((phi[0],phi[-1],theta[0],theta[-1]))*R2D )
        pylab.colorbar()

        pylab.subplot(7,1,5)
        pylab.title('undelayed')
        axim_undelayed = pylab.imshow( ni, vmin=vmin, vmax=vmax,
                                     extent=np.array((phi[0],phi[-1],theta[0],theta[-1]))*R2D )
        pylab.colorbar()

        vmin=-(255.0**2)
        vmax=255.0**2
        pylab.subplot(7,1,6)
        pylab.title('EMDs')
        axim_emds = pylab.imshow( emds, vmin=vmin, vmax=vmax )
        pylab.colorbar()

        #pylab.figure()
        ax_Vm = pylab.subplot(7,1,7)
        pylab.draw()

    cur_time = 0.0
    times = []
    Vms = []

    pixshift = 0.0
    while cur_time < duration_sec:
        if (cur_time % 0.1) < (1.0/hz):
            print cur_time,'of',duration_sec
            
        pixshift += vel_pix_per_dt
        nd_image.shift( i2, (0.0, pixshift), mode='wrap', output=shifted, order=1 )
        ni = im2r(shifted)

        # Arbitrary gain change to match Lindemann's compartment model.
        # (With this gain, the conductances roughly match his.)
        ni = ni*5.5

        emds = emd_sim.step( np.ravel(ni) )

        subunit_A_Bd = emd_sim.get_values('subunit_A_Bd')
        subunit_Ad_B = emd_sim.get_values('subunit_Ad_B')
        
        if 0:
            print 'np.sum(np.ravel(ni))',np.sum(np.ravel(ni))
            for n in ['earlyvis','delayed','undelayed','subunit_A_Bd']:
                print n,np.sum(np.ravel(emd_sim.get_values(n)))
            # rectify
            re = np.where( subunit_A_Bd > 0, subunit_A_Bd , 0 )
            ri = np.where( subunit_Ad_B > 0, subunit_Ad_B , 0 )

            print 'np.nonzero(re).shape',np.nonzero(re).shape
            ge = np.sum(weights_A*re)
            gi = np.sum(weights_B*ri)
            print 'ge',ge
            print 'gi',gi
            print
            
        Vm = compartment.step( subunit_A_Bd, subunit_Ad_B )
        #Vm = compartment.step( subunit_Ad_B, subunit_A_Bd )

        # save conductance for full compartmental model
        row = saver.row
        row['cur_time'] = cur_time
        row['directionA'] = np.asarray( subunit_A_Bd )
        row['directionB'] = np.asarray( subunit_Ad_B )
        row['EMD_outputs'] = np.asarray( emds )
        row['Vm'] = Vm
        row.append()
        saver.flush()

        cur_time += 1.0/hz

        times.append( cur_time )
        Vms.append( Vm )
        
        if DBG:
            emds.shape = emds_shape
        
            axim_shifted.set_array( shifted )
            axim_ni.set_array( ni )
            axim_emds.set_array( emds )

            axim_earlyvis.set_array( emd_sim.get_values('earlyvis', ni.shape ))
            axim_delayed.set_array( emd_sim.get_values('delayed', ni.shape ))
            axim_undelayed.set_array( emd_sim.get_values('undelayed', ni.shape ))

            ax_Vm.plot( times, Vms )
            
            pylab.draw()
    h5file.close()
    return Vms


if __name__=='__main__':

    imfiles = glob.glob('stim_images_fullsize/*')
    imfile = imfiles[0]
    vel = -200.0
    Vms = doit( (imfile, vel) )

    if DBG:
        pylab.show()
        


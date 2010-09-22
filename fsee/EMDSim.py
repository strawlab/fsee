# Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
from __future__ import division
import numpy as nx
import numpy
import scipy
import scipy.signal as signal
import warnings
import sys

__all__ = ['prony','FilterMaker','EMDSim','SingleCompartmentSim',
           'log_normal', 'unity_gain_log_normal',
           'get_complete_log_normal_curve', 'get_smallest_filter_coefficients']

def prony(h, nb, na):
    """Prony's method for time-domain IIR filter design.

    Description:

      Finds a filter with numerator order na, denominator order nb,
      and having the impulse response in array h.  The IIR filter
      coefficients are returned in length nb+1 and na+1 row vectors b
      and a, ordered in descending powers of Z.

    Inputs:

      h --- impulse response to fit
      nb, na -- number of filter coefficients

    Outputs:

      b,a -- Numerator and denominator of the iir filter.
    """

    zeros = nx.zeros
    transpose = nx.transpose
    toeplitz = scipy.linalg.toeplitz
    hstack = nx.hstack
    vstack = nx.vstack
    matrixmultiply = nx.dot
    newaxis = nx.newaxis

    lstsq = scipy.linalg.lstsq

    h = nx.asarray(h)
    K = len(h) - 1
    M = nb
    N = na
    if K < max(M,N):
        raise ValueError('Model order too large for data. Zero-pad data to fix?')
    c = h[0]
    if c==0: # avoid divide by zero
        c=1
    row = zeros((K+1,))
    row[0] = (h/c)[0] # avoid scipy warning
    H = toeplitz(h/c,row)
    if K > N:
        H = H[:,:N+1]
    # Partition H matrix
    H1 = H[:(M+1),:]
    h1 = H[(M+1):,:1]
    H2 = H[(M+1):,1:]

    x,resids,rank,s = lstsq(-H2,h1)
    a = vstack(([1],x))[:,0]
    b = matrixmultiply(H1,c*a[:,newaxis])[:,0]
    return b,a

# See COOMBE, PE "THE LARGE MONOPOLAR CELLS L1 AND L2 ARE RESPONSIBLE
# FOR ERG TRANSIENTS IN DROSOPHILA"

def log_normal(t,K=1.0,tp=0.020,sigma=0.355):
    # Log normal model from Payne, R., & Howard, J. (1981). Response
    # of an Insect Photoreceptor - a Simple Log-Normal Model. Nature,
    # 290 (5805), 415-416.

    # See also Howard, J., Dubs, A., & Payne, R. (1984) Dynamics of
    # phototransduction in insects. Journal of Comparative Physiology
    # A, 154, 707-718.

    # Coefficients default values fit from data in Juusola, M., &
    # Hardie, R.C. (2001). Light adaptation in Drosophila
    # photoreceptors: I. Response dynamics and signaling efficiency at
    # 25 degrees C. Journal of General Physiology, 117 (1), 3-25.
    # http://www.jgp.org/cgi/content/full/117/1/3
    # doi:10.1085/jgp.117.1.3

    """

V(t)=\exp \left[ -\frac{ \left( \log \frac{t}{t_p}
\right)^2}{2\sigma^2} \right]

    """
    return K*nx.exp(-(nx.log(t/tp)**2)/(2*sigma**2))

def unity_gain_log_normal(t,tp=0.020,sigma=0.355,dt=1.0):
    integral = nx.exp( sigma**2/2.0 )*nx.sqrt(2*nx.pi)*nx.sqrt(sigma**2)*tp
    return log_normal(t,tp=tp,sigma=sigma)/integral*dt

def get_complete_log_normal_curve(tp=0.020,sigma=0.355,dt=1.0, eps=1e-15, max_tsteps=1e6):
    maxt = tp
    while 1:
        t = nx.arange(0,maxt,dt,dtype=nx.float64)
        if t.shape[0] > max_tsteps:
            raise ValueError('more timesteps needed than max_tsteps')
        V = unity_gain_log_normal( t, tp=tp, sigma=sigma, dt=dt )
        if V[-1] < eps:
            break
        maxt = 2*maxt
    return V

def get_smallest_filter_coefficients( impulse_response ):
    """get b,a that recreate impulse response"""

    # step 1. Calculate smallest set of filter coefficients that
    # accurately recreates impulse response (nb==na).

    last_good = False
    nba = 20
    input = nx.zeros(impulse_response.shape,dtype=nx.float64)
    input[0]=1 # impulse
    while 1:
        b,a = prony(impulse_response,nba,nba)
        testV = signal.lfilter(b,a,input)
        if last_good:
            if not nx.allclose(impulse_response,testV):
                nba = last_good_nba
                b,a = prony(impulse_response,nba,nba)
                break
            else:
                last_good_nba = nba
                nba -= 1
        else:
            if nx.allclose(impulse_response,testV):
                last_good_nba = nba
                last_good = True
            else:
                nba += 1

    # step 2. Calculate smallest a possible
    na = nba -1
    while 1:
        b,a = prony(impulse_response,nba,na)
        testV = signal.lfilter(b,a,input)
        if nx.allclose(impulse_response,testV):
            na -= 1
        else:
            na += 1
            break

    # step 3. Calculate smallest b possible
    nb = nba -1
    while 1:
        b,a = prony(impulse_response,nb,nba)
        testV = signal.lfilter(b,a,input)
        if nx.allclose(impulse_response,testV):
            nb -= 1
        else:
            nb += 1
            break

    # step 4. Return smallest set of filter coefficients possible
    if nb < na:
        nb = nb
        na = nba
    else:
        nb = nba
        na = na
    b,a = prony(impulse_response,nb,na)
    return b,a

def test_lognormal():
    hz=2000.0
    t=nx.arange(0,1,1/hz)

    import pylab
    sigma = 0.355
    tp = 0.02
    y = unity_gain_log_normal(t,tp=tp,sigma=sigma,dt=1.0/hz)

    V = get_complete_log_normal_curve(tp=tp,sigma=sigma,dt=1.0/hz)
    b,a = get_smallest_filter_coefficients( V )

    input = nx.zeros(t.shape,dtype=nx.float64)
    input[0]=1 # impulse
    testV = signal.lfilter(b,a,input)
    pylab.plot(t,y,'bx')
    pylab.plot(t,testV,'r.')
    pylab.show()

def compose_transfer_functions(ba0,ba1):
    b0=ba0[0]
    b1=ba1[0]
    a0=ba0[1]
    a1=ba1[1]
    b = scipy.polymul(b0,b1)
    a = scipy.polymul(a0,a1)
    return (b,a)

class FilterMaker:
    def __init__(self,hz):
        self.hz = float(hz)
        self.unity_gain = True
    def iir_lowpass1(self,tau=0.008,mult=1.0,analytic=False):
        """first order low-pass IIR filter"""
        dt = 1.0/self.hz

        if analytic:
            # XXX I should really do this with c2d and make sure it's OK
            if mult != 1.0:
                raise NotImplementedError('')
            b = nx.array([dt/tau])
            a = nx.array([1,b[0]-1])
            return b,a
        else:
            max_t = tau*20
            t = nx.arange(0,max_t+dt,dt)
            V = 1.0/tau * nx.exp(-t/tau)
            if self.unity_gain:
                V = V/abs(nx.sum(V))
            V*=mult
            b,a = prony(V,0,1) # this is, by definition 1st order filter
            return b,a
    def fir_lowpass1(self,tau=0.008,mult=1.0):
        """first order low-pass FIR filter"""
        dt = 1.0/self.hz
        max_t = tau*20
        t = nx.arange(0,max_t+dt,dt)
        V = -1.0/tau * nx.exp(-t/tau)
        if self.unity_gain:
            V = V/abs(nx.sum(V))
        V*=mult
        b=V
        a=[1.0]
        return b,a
    def iir_highpass1(self,tau=0.5,mult=1.0,analytic=False):
        """first order high-pass IIR filter"""
        dt = 1.0/self.hz
        if analytic:
            if mult != 1.0:
                raise NotImplementedError('')
            T = dt/tau
            T1 = T-1
            b = nx.array([-T1,T1])
            a = nx.array([1,T1])
            return b,a
        else:
            max_t = tau*20
            t = nx.arange(0,max_t+dt,dt)
            V = -1.0/tau * nx.exp(-t/tau)
            if self.unity_gain:
                V = V/abs(nx.sum(V))
            V[0] = V[0]+1.0 # make highpass
            V*=mult
            b,a = prony(V,1,1) # this is, by definition 1st order filter
            return b,a
    def fir_highpass1(self,tau=0.5,mult=1.0):
        """first order high-pass FIR filter"""
        dt = 1.0/self.hz
        max_t = tau*20
        t = nx.arange(0,max_t+dt,dt)
        V = -1.0/tau * nx.exp(-t/tau)
        if self.unity_gain:
            V = V/abs(nx.sum(V))
        V[0] = V[0]+1.0 # make highpass
        V*=mult
        b=V
        a=[1.0]
        return b,a
    def james_lmc(self,
                  a1=-1.06,tau1=0.012,sigma1=0.197,
                  a2=0.167,tau2=0.021,sigma2=0.345):
        # see Lindemann, et al. 2005

        # XXX With default parameters, this is not a perfect high pass
        # and has a lowpass gain of approximately -0.003.
        dt = 1.0/self.hz
        max_t = .5
        t = nx.arange(0,max_t+dt,dt)
        V=a1*nx.exp(-(nx.log(t/tau1))**2/(2*sigma1**2))+a2*nx.exp(-(nx.log(t/tau2))**2/(2*sigma2**2))
        V = V/self.hz # minimize gain dependence on sample frequency
        eps = 1e-16
        assert abs(V[-1])<eps # make sure it's sampled long enough
        b,a = prony(V,14,10)
        if 1:
            # ensure order of filter is high enough
            input = nx.zeros(t.shape,nx.float64)
            input[0] = 1 # impulse response
            output=signal.lfilter(b,a,input)
            testlen = min(len(output),len(V))
            assert nx.allclose(output[:testlen],V[:testlen])
        return b,a

class EMDSim:
    def __init__(self,

                 earlyvis_ba = None, # early vision temporal filters, if None, set to Drosophila estimates
                 early_contrast_saturation_params=None, # if None, don't do any contrast saturation
                 emd_lp_ba = None, # delay filter of EMD, if None set to 35 msec
                 emd_hp_ba = None, # highpass tau (can be None)

                 subtraction_imbalance = 1.0, # 1.0 for perfect subtraction
                 lindemann_weight_map = None,
                 compute_typecode = nx.float32,
                 hz=200.0,
                 n_receptors=60,
                 emd_edges=None, # list of tuples [(A1,B1),(A2,B2),(A3,B3)]
                 sign_convention=1, # 0 or 1, arbitrary

                 # further processing control
                 do_luminance_adaptation=False,

                 preEMD_saturation_s=None,
                 # Note that the implementation of preEMD_saturation
                 # is mathematically equivalent to
                 # early_contrast_saturation_params.
                ):
        if emd_edges is None:
            emd_edges = []

        self.sign_convention = sign_convention

        self.compute_typecode = compute_typecode
        del compute_typecode
        self.n_receptors = n_receptors
        del n_receptors

        self.emd_sideA_idxs = nx.asarray( [e[0] for e in emd_edges] )
        self.emd_sideB_idxs = nx.asarray( [e[1] for e in emd_edges] )
        del emd_edges

        if earlyvis_ba is None:
            if 1:
                #print 'IMPORTANT: using photoreceptor values for Drosophila'
                sys.stderr.write("fsee: EMDsim.py: IMPORTANT: using photoreceptor temporal dynamics " + 
                      "values for Drosophila fit to Juusola & Hardie, 2001\n");
                tp = 0.02
                sigma = 0.355

                # It would be nice to do a Laplace transform of the
                # "log normal" and then convert that to a discrete
                # time representation, but mathematica couldn't find
                # the Laplace transform of that function.

                # Upon further research, it appears that "log normal"
                # as defined by the insect vision community, is
                # different than what the statistics community
                # uses. Also, any arbitrary function can be
                # represented as an nth order ODE through the use of
                # Prony's method. So, we could find the Laplace
                # transform using that.

                V = get_complete_log_normal_curve(tp=tp,sigma=sigma,dt=1.0/hz)
                self.b_earlyvis, self.a_earlyvis = get_smallest_filter_coefficients(V)

            else:
                sys.stderr.write('fsee: EMDSim.py: IMPORTANT: using photoreceptor values for Eristalis\n')
                if hz != 200.0:
                    raise ValueError('Photoreceptor fits for 200 Hz. Use fit_photoreceptor')

                self.b_earlyvis = nx.array([ 0.,  0.00275785,  0.44602765,  0.66420313],
                                             self.compute_typecode)
                self.a_earlyvis = nx.array([ 1., -0.75061758,  0.20058061],
                                             self.compute_typecode)
        else:
            self.b_earlyvis = nx.asarray(earlyvis_ba[0]).astype(self.compute_typecode)
            self.a_earlyvis = nx.asarray(earlyvis_ba[1]).astype(self.compute_typecode)

        self.do_luminance_adaptation = do_luminance_adaptation
        if self.do_luminance_adaptation not in [True,False]:
            raise ValueError('do_luminance_adaptation must be True or False')
        if self.do_luminance_adaptation: # luminance adaptation
            tmpfm=FilterMaker(hz)
            tmptau=.1#5.0
            sys.stderr.write('fsee: EMDSim.py: Using luminance adaptation with 1st-order '
                   'highpass, tau = %f seconds.\n'%tmptau)
            self.b_lum_adapt,self.a_lum_adapt = tmpfm.iir_highpass1(tau=tmptau,analytic=True)
            del tmpfm
            del tmptau
        else:
            sys.stderr.write("fsee: EMDsim.py: Not using EMDSim.py's quick luminance adaptation\n")
        self._luminance_adapted = None

        self.early_contrast_saturation_params = early_contrast_saturation_params

        self.preEMD_saturation_s=preEMD_saturation_s
        self._D_pre_saturation = None
        self._U_pre_saturation = None

        if len(self.b_earlyvis)==1 and self.b_earlyvis[0]==1.0 and len(self.a_earlyvis)==0:
            self.skip_earlyvis = True
            warnings.warn('skipping early visual simulation '
                          'because scipy.signal.lfilter cannot '
                          'handle b=[1.0],a=[]') # fix one day!
        else:
            self.skip_earlyvis = False

        if emd_hp_ba is not None:
            self.do_highpass = True
            # highpass filter
            self.b_hp, self.a_hp=emd_hp_ba
            self.b_hp = numpy.asarray(self.b_hp,dtype=self.compute_typecode)
            self.a_hp = numpy.asarray(self.a_hp,dtype=self.compute_typecode)
        else:
            self.do_highpass = False

        # EMD (lowpass) filter
        if emd_lp_ba is None:
            tau_emd = 0.035
            fm = FilterMaker(hz)
            sys.stderr.write('fsee: EMDsim.py: making EMD lowpass filter: tau ~%d msec\n'%int(tau_emd*1000.0))
            self.b_emd, self.a_emd=fm.iir_lowpass1(tau_emd)
        else:
            self.b_emd, self.a_emd=emd_lp_ba
##        self.b_emd = numpy.asarray(self.b_emd,dtype=self.compute_typecode)
##        self.a_emd = numpy.asarray(self.a_emd,dtype=self.compute_typecode)
        self.b_emd = numpy.array(self.b_emd,dtype=self.compute_typecode)
        self.a_emd = numpy.array(self.a_emd,dtype=self.compute_typecode)

        # compute filter coefficients for each channel

        if self.do_luminance_adaptation:
            self.zi_luminance_adaptation = None # set later

        if not self.skip_earlyvis:
            z0_earlyvis = nx.zeros( (max(len(self.a_earlyvis),len(self.b_earlyvis))-1,), self.compute_typecode )
            self.zi_earlyvis = nx.resize( z0_earlyvis, (self.n_receptors,len(z0_earlyvis)) ) # repeat to fill array

        if self.do_highpass:
            z0_hp = nx.zeros( (max(len(self.a_hp),len(self.b_hp))-1,), self.compute_typecode )
            self.zi_hp = nx.resize( z0_hp, (self.n_receptors,len(z0_hp)) )

        z0_nolmc_emd = nx.zeros( (max(len(self.a_emd),len(self.b_emd))-1,), self.compute_typecode )
        self.zi_nolmc_emd = nx.resize( z0_nolmc_emd, (self.n_receptors,len(z0_nolmc_emd)) )

        z0_emd = nx.zeros( (max(len(self.a_emd),len(self.b_emd))-1,), self.compute_typecode )
        self.zi_emd = nx.resize( z0_emd, (self.n_receptors,len(z0_emd)) )

        if lindemann_weight_map is None:
            weightmap = nx.ones( (self.n_receptors,), self.compute_typecode )
        else:
            weightmap = nx.asarray( lindemann_weight_map ).astype( self.compute_typecode )

        self.weights_A = weightmap[self.emd_sideA_idxs]
        self.weights_B = weightmap[self.emd_sideB_idxs]

        self.weights_A = self.weights_A[:,nx.newaxis]
        self.weights_B = self.weights_B[:,nx.newaxis]

        # subtraction imbalance
        self.S = nx.array(subtraction_imbalance,self.compute_typecode)

        self.emd_outputs = None

        self._earlyvis = None
        self._D = None
        self._U = None
        self._subunit_A_Bd = None
        self._subunit_Ad_B = None

    def step(self,responses):
        retinal_image = nx.asarray( responses )
        retinal_image = retinal_image.astype( self.compute_typecode )
        assert retinal_image.shape == (self.n_receptors,)

        self._retinal_image = retinal_image[:,nx.newaxis] # we operate on rank-2 arrays

        if self.do_luminance_adaptation:
            if self.zi_luminance_adaptation is None:

                # This is the first step, so find filter coefficients
                # that produce zero output to produce perfectly
                # adapted filter state.
                y = nx.zeros_like(self._retinal_image)
                x = self._retinal_image

                n_elements_state_vec = max(len(self.b_lum_adapt),len(self.b_lum_adapt))-1
                zi_shape = (self.n_receptors,n_elements_state_vec)

                if 0:
                    self.zi_luminance_adaptation = signal.lfiltic(
                        self.b_lum_adapt, self.a_lum_adapt, y, x, axis=1)
                else:
                    self.zi_luminance_adaptation = nx.zeros( zi_shape,
                                                             self.compute_typecode )
                    for i in range(self.n_receptors):
                        this_zi = signal.lfiltic(
                            self.b_lum_adapt, self.a_lum_adapt, y[i,:], x[i,:])
                        self.zi_luminance_adaptation[i,:] = this_zi.astype(
                            self.compute_typecode)
                del y
                del x

                if zi_shape != self.zi_luminance_adaptation.shape:
                    print 'wanted shape %s, got shape %s'%(
                        str(zi_shape),str(self.zi_luminance_adaptation.shape))
                    raise ValueError('shape wrong')

                test_zero, tmpzi = signal.lfilter(self.b_lum_adapt,
                                                  self.a_lum_adapt,
                                                  self._retinal_image,
                                                  axis=1,
                                                  zi=self.zi_luminance_adaptation)
                epsilon = 1e-5
                if test_zero.max() > epsilon:
                    raise ValueError("maximum value shouldn't be greater than epsilon")

            (self._luminance_adapted,
             self.zi_luminance_adaptation) = signal.lfilter(self.b_lum_adapt,
                                                            self.a_lum_adapt,
                                                            self._retinal_image,
                                                            axis=1,
                                                            zi=self.zi_luminance_adaptation)
            #print 'set self._luminance_adapted'
        else:
            self._luminance_adapted = self._retinal_image

        # early vision (photoreceptor/LMC) filtering
        if not self.skip_earlyvis:
            self._earlyvis, self.zi_earlyvis = signal.lfilter(self.b_earlyvis,
                                                              self.a_earlyvis,
                                                              self._luminance_adapted,
                                                              axis=1,
                                                              zi=self.zi_earlyvis)
        else:
            self._earlyvis = self._retinal_image

        if self.early_contrast_saturation_params is not None:
            tmp = self.early_contrast_saturation_params
            csat_type = tmp[0]
            if csat_type == 'tanh+lin':
                a, b = self.early_contrast_saturation_params[1:]
                self._early_contrast_saturated = numpy.tanh( self._earlyvis * a) + self._earlyvis*b
            elif csat_type == 'tanh':
                a = self.early_contrast_saturation_params[1]
                self._early_contrast_saturated = numpy.tanh( self._earlyvis * a)
            else:
                raise ValueError('unknown contrast saturation type: %s'%csat_type)
        else:
            self._early_contrast_saturated = self._earlyvis

        # high pass filter if necessary
        if self.do_highpass:
            self._U, self.zi_hp = signal.lfilter(self.b_hp,
                                                 self.a_hp,
                                                 self._early_contrast_saturated,
                                                 axis=1,
                                                 zi=self.zi_hp)
        else:
            self._U = self._early_contrast_saturated # undelayed is just early vision filtering

        # emd lowpass filter
        self._D, self.zi_emd = signal.lfilter(self.b_emd,
                                              self.a_emd,
                                              self._U,axis=1,
                                              zi=self.zi_emd)

        self._U_pre_saturation = self._U
        self._D_pre_saturation = self._D
        if self.preEMD_saturation_s is not None:
            # compression/saturation, a la Dror. 2001, eqn. 5
##            sU = self.preEMD_saturation_s*self._U
##            self._U = nx.tanh(sU)
##            print sU[:5],'->',self._U[:5]
            self._U = nx.tanh(self.preEMD_saturation_s*self._U)
            self._D = nx.tanh(self.preEMD_saturation_s*self._D)

        # half correlators
        # A * Bdelayed
        self._subunit_A_Bd = self._U[self.emd_sideA_idxs] * self._D[self.emd_sideB_idxs]
        # Adelayed * B
        self._subunit_Ad_B = self._D[self.emd_sideA_idxs] * self._U[self.emd_sideB_idxs]

        # flicker insensitive
        if self.sign_convention:
            self.emd_outputs = (self.weights_A*self._subunit_A_Bd -
                                self.S*self.weights_B*self._subunit_Ad_B)
        else:
            self.emd_outputs = (self.S*self.weights_B*self._subunit_Ad_B -
                                self.weights_A*self._subunit_A_Bd)

        return self.emd_outputs[:,0] # make rank-1

    def get_values(self, val_name, shape=None):
        n2i = {'luminance_adapted':self._luminance_adapted,
               'earlyvis':self._earlyvis,
               'early_contrast_saturated':self._early_contrast_saturated,
               'delayed':self._D,
               'undelayed':self._U,
               'delayed_pre_saturation':self._D_pre_saturation,
               'undelayed_pre_saturation':self._U_pre_saturation,
               'subunit_A_Bd':self._subunit_A_Bd,
               'subunit_Ad_B':self._subunit_Ad_B,
               'weights_A':self.weights_A,
               'weights_B':self.weights_B,
               'emd_outputs':self.emd_outputs
               }
        valarr = n2i[val_name]
        if shape is None:
            return valarr[:,0] # make rank-1
        else:
            return nx.reshape( valarr, shape )

class SingleCompartmentSim:
    def __init__(self,weights_A,weights_B,lp_BA,Ee=50.0):
        self.weights_A = weights_A
        self.weights_B = weights_B
        self.lp_B, self.lp_A = lp_BA
        self.compute_typecode = nx.float32

        # compute filter coefficients
        z0 = nx.zeros( (max(len(self.lp_A),len(self.lp_B))-1,), self.compute_typecode )
        self.zi = nx.resize( z0, (1,len(z0)) )

        self.g0 = 1295
        self.Ee = Ee
        self.Ei_gain = -0.95

    def step(self,subunit_A_Bd,subunit_Ad_B):
        # excitatory direction is A * Bd (arbitrary)

        # rectify
        re = nx.where( subunit_A_Bd > 0, subunit_A_Bd , 0 )
        ri = nx.where( subunit_Ad_B > 0, subunit_Ad_B , 0 )

        ge = nx.sum(self.weights_A*re)
        gi = nx.sum(self.weights_B*ri)

        # print 'ge + gi',ge + gi # XXX this should be near 1600 to
        # match Lindemann et al. 2005

        E0 = 0
        Ei = self.Ei_gain*self.Ee

        Vm_fast = (self.g0*E0 + ge*self.Ee + gi*Ei)/(self.g0+ge+gi)

        Vm_fast_array = nx.array(Vm_fast,self.compute_typecode)
        Vm_fast_array.shape = 1,1 # make rank 2

        Vm_slow, self.zi = signal.lfilter(self.lp_B,
                                          self.lp_A,
                                          Vm_fast_array,
                                          axis=1,
                                          zi=self.zi)
        return float(Vm_slow[0,0])
        #return float(Vm_fast_array[0,0])

##if __name__=='__main__':
##    test_lognormal()

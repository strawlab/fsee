import os, math
import pickle

import cgtypes

import fsee
import fsee.Observer
import fsee.EMDSim

import pylab

import scipy
import scipy.io

data=scipy.io.loadmat('alice_data')
d2=data['DATA']

x=d2[0,:]

y=d2[1,:]

D2R = math.pi/180.0
theta=d2[2,:]*D2R

xoffset = 753
yoffset = 597

radius_pix = 753-188 # pixels

# 10 inch = 25.4 cm = 254 mm = diameter = 2*radius
pix_per_mm = 2.0*radius_pix/254.0

mm_per_pixel = 1.0/pix_per_mm

xgain = mm_per_pixel
ygain = mm_per_pixel

x_mm=(x-xoffset)*xgain
y_mm=(y-yoffset)*ygain

PLOT_RECEPTORS = True
if PLOT_RECEPTORS:
    import fsee.plot_utils

def main():
    hz = 200.0
    tau_emd = 0.1

    # get IIR filter coefficients for tau and sample freq
    emd_lp_ba = fsee.EMDSim.FilterMaker(hz).iir_lowpass1(tau_emd)

    pos_vec3 = cgtypes.vec3( 0, 0, 0)
    #ori_quat = cgtypes.quat( 1, 0, 0, 0)
    #ori_quat = cgtypes.quat().fromAngleAxis( math.pi*0.5, (0,0,1) )

    model_path = os.path.join(fsee.data_dir,"models/alice_cylinder/alice_cylinder.osg")
    vision = fsee.Observer.Observer(model_path=model_path,
                                    scale=1000.0, # make cylinder model 1000x bigger
                                    hz=hz,
                                    skybox_basename=None, # no skybox
                                    emd_lp_ba = emd_lp_ba,
                                    full_spectrum=PLOT_RECEPTORS,
                                    optics='buchner71',
                                    do_luminance_adaptation=False,
                                    )

    vel = 0.5 # meters/sec
    dt = 1/hz
    mean_emds = {}
    z = 2 # 2 mm
    posname = 'alice'
#    startpos = cgtypes.vec3( x_mm[0], y_mm[0], z))
    if 1:
##    for posname,startpos in [ ('left',cgtypes.vec3( 0, 110, 0)),
##                              #('center',cgtypes.vec3( 0, 0, 0)),
##                              #('right',cgtypes.vec3( 0, -110, 0)),
##                              ]:
        sum_vec = None
        tnow = 0.0
        #pos = startpos
        #posinc = cgtypes.vec3( vel*dt*1000.0, 0, 0) # mm per step
        #posnow = startpos
        count = 0
        while count < len(x_mm):
            ori_quat = cgtypes.quat().fromAngleAxis(theta[count],(0,0,1))
            posnow = cgtypes.vec3(( x_mm[count], y_mm[count], z))
            vision.step(posnow,ori_quat)
            if 1:
                #if (count-1)%20==0:
                if 1:
                    fname = 'alice_envmap_%04d.png'%(count,)
                    vision.save_last_environment_map(fname)
            EMDs = vision.get_last_emd_outputs()
            if PLOT_RECEPTORS:
                if (count-1)%20==0:
                    R=vision.get_last_retinal_imageR()
                    G=vision.get_last_retinal_imageG()
                    B=vision.get_last_retinal_imageB()
                    fsee.plot_utils.plot_receptor_and_emd_fig(
                        R=R,G=G,B=B,
                        emds=EMDs,
                        scale=5e-3,
                        figsize=(10,10),
                        dpi=100,
                        save_fname='receptors_%s_%04d.png'%(posname,count),
                        optics=vision.get_optics(),
                        proj='stere',
                        )

            if sum_vec is None:
                sum_vec = EMDs
            else:
                sum_vec += EMDs
            #posnow = posnow + posinc
            tnow += dt
            count += 1
        mean_emds[posname]= (sum_vec/count)

    fd = open('full3D_emds.pkl','wb')
    pickle.dump( mean_emds, fd )
    fd.close()

if 1:
    # don't profile
    main()

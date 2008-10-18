import os, math, time
import pickle

import cgtypes

import fsee
import fsee.Observer
import fsee.EMDSim

import pylab

import scipy
import scipy.io

def main():

    if 1:
        # load some data from Alice's arena
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

    hz = 200.0
    tau_emd = 0.1

    # get IIR filter coefficients for tau and sample freq
    emd_lp_ba = fsee.EMDSim.FilterMaker(hz).iir_lowpass1(tau_emd)

    model_path = os.path.join(fsee.data_dir,"models/alice_cylinder/alice_cylinder.osg")
    vision = fsee.Observer.Observer(model_path=model_path,
                                    scale=1000.0, # make cylinder model 1000x bigger (put in mm instead of m)
                                    hz=hz,
                                    skybox_basename=None, # no skybox
                                    emd_lp_ba = emd_lp_ba,
                                    full_spectrum=True,
                                    optics='buchner71',
                                    do_luminance_adaptation=False,
                                    )

    dt = 1/hz
    z = 2 # 2 mm
    if 1:
        count = 0
        tstart = time.time()
        try:
            while count < len(x_mm):
                ori_quat = cgtypes.quat().fromAngleAxis(theta[count],(0,0,1))
                posnow = cgtypes.vec3(( x_mm[count], y_mm[count], z))
                vision.step(posnow,ori_quat)
                count += 1
        finally:
            tstop = time.time()
            dur = tstop-tstart
            fps = count/dur
            print '%d frames rendered in %.1f seconds (%.1f fps)'%(count,dur,fps)

if __name__=='__main__':
    if int(os.environ.get('PROFILE','0')):
        import cProfile
        import lsprofcalltree
        p = cProfile.Profile()
        p.run('main()')
        k = lsprofcalltree.KCacheGrind(p)
        data = open(os.path.expanduser('~/fsee_simple_benchmark.kgrind'), 'w+')
        k.output(data)
        data.close()
    else:
        main()

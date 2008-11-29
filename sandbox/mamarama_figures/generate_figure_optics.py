import os

import cgkit.cgtypes as cgtypes

import fsee
import fsee.Observer
import fsee.plot_utils
import pylab
import sys

pos_vec3 = cgtypes.vec3( 600, 400, 200)
ori_quat = cgtypes.quat( 1, 0, 0, 0)

model_path = os.path.join(fsee.data_dir,"models/mamarama_checkerboard/mamarama_checkerboard.osg")
optics = 'buchner71'
vision = fsee.Observer.Observer(model_path=model_path,
                                scale=1000.0, # convert model from meters to mm
                                hz=200.0,
                                skybox_basename=None,
                                full_spectrum=True,
                                optics=optics,
                                do_luminance_adaptation=False,
                                )
vision.step(pos_vec3,ori_quat)
#vision.save_last_environment_map('simple_save_envmap.png')

R=vision.get_last_retinal_imageR()
G=vision.get_last_retinal_imageG()
B=vision.get_last_retinal_imageB()
emds = vision.get_last_emd_outputs()
for fname in ['optics.png',
              #'optics.pdf',
              ]:
    fsee.plot_utils.plot_receptor_and_emd_fig(
        figsize=(3.6,2.25),
        R=R,G=G,B=B,#emds=emds,
        save_fname=fname,
        optics = optics,
        overlay_receptor_circles=True,
        dpi=300)
#pylab.show()

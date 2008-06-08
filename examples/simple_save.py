# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import os

import cgkit.cgtypes as cgtypes

import fsee
import fsee.Observer
import fsee.plot_utils

pos_vec3 = cgtypes.vec3( 0, 0, 0)
ori_quat = cgtypes.quat( 1, 0, 0, 0)

model_path = os.path.join(fsee.data_dir,"models/tunnel_leftturn/tunnel.osg")
vision = fsee.Observer.Observer(model_path=model_path,
                                hz=200.0,
                                full_spectrum=True,
                                optics='buchner71',
                                do_luminance_adaptation=False,
                                )
vision.step(pos_vec3,ori_quat)
vision.save_last_environment_map('simple_save_envmap.png')

R=vision.get_last_retinal_imageR()
G=vision.get_last_retinal_imageG()
B=vision.get_last_retinal_imageB()
emds = vision.get_last_emd_outputs()
fsee.plot_utils.plot_receptor_and_emd_fig(
    R=R,G=G,B=B,emds=emds,
    save_fname='simple_save.png',
    optics = vision.get_optics(),
    dpi=200)

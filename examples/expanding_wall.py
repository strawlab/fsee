# Copyright (C) 2010 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import os, math, warnings

import numpy as np

import cgtypes # cgkit 1.x

from matplotlib import rcParams

rcParams['svg.embed_char_paths'] = False
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial'] # lucid: ttf-mscorefonts-installer

font_size = 10
rcParams['axes.labelsize'] = font_size
rcParams['xtick.labelsize'] = font_size
rcParams['ytick.labelsize'] = font_size

import fsee
import fsee.Observer
import fsee.plot_utils
import numpy as np

D2R = np.pi/180.0


model_path=os.path.abspath('auto_scene_gen/expanding_wall.osg')
vision = fsee.Observer.Observer(model_path=model_path,
                                hz=200.0,
                                full_spectrum=True,
                                optics='buchner71',
                                do_luminance_adaptation=False,
                                skybox_basename=os.path.join(fsee.data_dir,'Images/osgviewer_cubemap/'),
                                )
if 1:
    angle = 30*D2R
    dist2 = -1.5
    dist3 = 0.5
    ext = 'svg'

    # view from fly position
    pos_vec3 = cgtypes.vec3(0,0,10.0)
    ori_quat = cgtypes.quat().fromAngleAxis( angle,(0,0,1))
    vision.step(pos_vec3,ori_quat)
    vision.save_last_environment_map('expanding_wall1.png')

    R=vision.get_last_retinal_imageR()
    G=vision.get_last_retinal_imageG()
    B=vision.get_last_retinal_imageB()
    #emds = vision.get_last_emd_outputs()
    fsee.plot_utils.plot_receptor_and_emd_fig(
        R=R,G=G,B=B,#emds=emds,
        save_fname='expanding_wall_flyeye1.%s'%ext,
        optics = vision.get_optics(),
        proj='stere',
        dpi=200)

    pos_vec3 = cgtypes.vec3(dist2*np.cos(angle),dist2*np.sin(angle),10.0)
    ori_quat = cgtypes.quat().fromAngleAxis( angle,(0,0,1))
    vision.step(pos_vec3,ori_quat)
    vision.save_last_environment_map('expanding_wall2.png')

    R=vision.get_last_retinal_imageR()
    G=vision.get_last_retinal_imageG()
    B=vision.get_last_retinal_imageB()
    #emds = vision.get_last_emd_outputs()
    fsee.plot_utils.plot_receptor_and_emd_fig(
        R=R,G=G,B=B,#emds=emds,
        save_fname='expanding_wall_flyeye2.%s'%ext,
        optics = vision.get_optics(),
        proj='stere',
        dpi=200)

    pos_vec3 = cgtypes.vec3(dist3*np.cos(angle),dist3*np.sin(angle),10.0)
    ori_quat = cgtypes.quat().fromAngleAxis( angle,(0,0,1))
    vision.step(pos_vec3,ori_quat)
    vision.save_last_environment_map('expanding_wall3.png')

    R=vision.get_last_retinal_imageR()
    G=vision.get_last_retinal_imageG()
    B=vision.get_last_retinal_imageB()
    #emds = vision.get_last_emd_outputs()
    fsee.plot_utils.plot_receptor_and_emd_fig(
        R=R,G=G,B=B,#emds=emds,
        save_fname='expanding_wall_flyeye3.%s'%ext,
        optics = vision.get_optics(),
        proj='stere',
        dpi=200)

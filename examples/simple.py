# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
import os

import cgkit.cgtypes as cgtypes

import fsee
import fsee.Observer

pos_vec3 = cgtypes.vec3( 0, 0, 0)
ori_quat = cgtypes.quat( 1, 0, 0, 0)

model_path = os.path.join(fsee.data_dir,"models/tunnel_leftturn/tunnel.osg")
vision = fsee.Observer.Observer(model_path=model_path,
                                hz=200.0,
                                optics='synthetic',
                                do_luminance_adaptation=False,
                                )
vision.step(pos_vec3,ori_quat)

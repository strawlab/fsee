# Copyright (C) 2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import os, math, warnings

import numpy as np

import cgkit.cgtypes as cgtypes

import fsee
import fsee.Observer

# these numbers specify body position and wing position (currently only body is used)

nums = """-63.944819191780439 0.95360065664418736 -1.5677184054558335
0.97592747002588576 0.00013135157421694402 0.21809381132492375
0.00080340363848455045 -64.511912229419792 2.4958009855663787
-0.91416847614060104 0.83348670845615658 -0.079986858311146714
0.50516881885673259 0.2090609331733887 -64.507322639730816
-0.59116643445214989 -0.91599543302231212 0.20749684980883201
0.50542162231080134 -0.079722030902706131 0.83374962597225588"""

nums = map(float,nums.split())

pos_vec3,ori_quat = cgtypes.vec3(nums[0:3]),cgtypes.quat(nums[3:7])
M = cgtypes.mat4().identity().translate(pos_vec3)
M = M*ori_quat.toMat4()
# grr, I hate cgtypes to numpy conversions!
M = np.array((M[0,0],M[1,0],M[2,0],M[3,0],
              M[0,1],M[1,1],M[2,1],M[3,1],
              M[0,2],M[1,2],M[2,2],M[3,2],
              M[0,3],M[1,3],M[2,3],M[3,3]))
M.shape=(4,4)

model_name='models/tunnel_straight/tunnel_straight_small_texture.osg'
model_path = os.path.join(fsee.data_dir,model_name)
vision = fsee.Observer.Observer(model_path=model_path,
                                scale=1000.0,
                                hz=200.0,
                                full_spectrum=True,
                                optics='buchner71',
                                do_luminance_adaptation=False,
                                #skybox_basename=None,
                                )
if 1:
    # view of fly position from slightly behind fly
    vision.sim.insert_flybody()
    vision.sim.set_flytransform(M)

    ori = ori_quat
    pos = pos_vec3

    ahead_fly = cgtypes.quat(0,1,0,0) # fly coord system of straight ahead
    flylengths_follow=2
    center_offset=cgtypes.vec3( 0, 0, 0 )
    lookat_offset=cgtypes.vec3( 0, 0, 0 )

    ahead_world = ori*ahead_fly*ori.inverse()
    ahead_world_vec = cgtypes.vec3( ahead_world.x, ahead_world.y, ahead_world.z)
    if 1:
        # remove z component
        ahead_world_vec[2] = 0
        ahead_world_vec = ahead_world_vec.normalize()

    center = pos + -1*flylengths_follow*ahead_world_vec + center_offset
    lookat = pos + lookat_offset
    viewdir = (lookat-center).normalize()

    camera_yaw_from_pos_y = math.atan2( -viewdir.x, viewdir.y )
    camera_pitch_from_neg_z = math.acos( -viewdir.z )
    #camera_pitch_from_neg_z = 2*math.pi/8
    view_ori_vec3 = (cgtypes.quat().fromAngleAxis(camera_yaw_from_pos_y,(0,0,1))*
                     cgtypes.quat().fromAngleAxis(camera_pitch_from_neg_z,(1,0,0))
                     )

    # I should really implement a save_image() function that can use a
    # different camera than the fly-eye view camera. For now, though,
    # this is what we've got:
    warnings.warn('view direciton wrong')
    view_ori_quat = cgtypes.quat(1,0,0,0)
    vision.step(center, view_ori_quat)
    vision.save_last_environment_map('situated_fly.png')
else:
    # view from fly position
    vision.step(pos_vec3,ori_quat)
    vision.save_last_environment_map('situated_fly.png')

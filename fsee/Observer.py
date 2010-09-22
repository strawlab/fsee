# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw

import cgtypes # cgkit 1.x
import fsee
from fsee.CoreVisualSystem import CoreVisualSystem
from fsee.eye_geometry.projected_eye_coords import RapidPlotter
import fsee.eye_geometry.switcher

import os
if int(os.environ.get('FSEE_MULTIPROCESS','0')):
    import fsee.FlySimWrapFSOI as Simulation
else:
    import fsee.RealFSOI as Simulation

import numpy

class Observer: # almost a sub-class of CoreVisualSystem
    """integrates visual system simulation with OpenSceneGraph

    This is the 2nd generation, which can calculate retinal velocities.

    It is almost a sub-class of CoreVisualSystem.
    """
    def __init__(self,
                 # OpenSceneGraph parameters
                 model_path=None,
                 scale=1.0,
                 zmin=0.001,
                 zmax=1e10,
                 cuberes=64,

                 # EMD parameters
                 hz=200.0,
                 earlyvis_ba = None, # early vision filter (can be None)
                 early_contrast_saturation_params=None,
                 emd_lp_ba = None, # EMD arm 2
                 emd_hp_ba = None, # EMD arm 1 (can be None)
                 subtraction_imbalance = 1.0,
#                 mean_luminance_value = 127.5,

                 # other simulation parameters
                 skybox_basename=fsee.default_skybox,
                 full_spectrum=False,

                 optics=None,
                 do_luminance_adaptation=None,
                 plot_realtime=False
                 ):
        # AC: This is initialized only if we want to use it (later)
        # self.rp = RapidPlotter(optics=optics)
        self.rp = None
        self.plot_realtime = plot_realtime
        
        self.optics = optics
        self.cubemap_face_xres=cuberes
        self.cubemap_face_yres=cuberes

        self.sim = Simulation.Simulation(
            model_path=model_path,
            scale=scale,
            skybox_basename=skybox_basename,
            zmin=zmin,
            zmax=zmax,
            xres=self.cubemap_face_xres,
            yres=self.cubemap_face_yres,
            )


##        self.nodes = [world_node,
##                      self.sim.get_root_node(),
##                      self.sim.get_skybox_node(),
##                      ]
##        self.nodes2names = {world_node:'world',
##                            self.sim.get_root_node():'root',
##                            self.sim.get_skybox_node():'skybox',
##                            }
##        self.nodes2vels = {world_node:cgtypes.vec3(0,0,0),
##                           self.sim.get_root_node():cgtypes.vec3(0,0,0),
##                           self.sim.get_skybox_node():None, # no relative translation
##                           }

        self.cvs = CoreVisualSystem(hz=hz,
                                    cubemap_face_xres=self.cubemap_face_xres,
                                    cubemap_face_yres=self.cubemap_face_yres,
                                    earlyvis_ba = earlyvis_ba,
                                    early_contrast_saturation_params=early_contrast_saturation_params,
                                    emd_lp_ba = emd_lp_ba,
                                    emd_hp_ba = emd_hp_ba,
                                    subtraction_imbalance=subtraction_imbalance,
                                    debug=False,
                                    full_spectrum=full_spectrum,
                                    #mean_luminance_value = mean_luminance_value,
                                    optics=optics,
                                    do_luminance_adaptation=do_luminance_adaptation,
                                    )
        precomputed = fsee.eye_geometry.switcher.get_module_for_optics(optics=optics)
        for attr in ['get_last_emd_outputs',
                     'get_last_retinal_imageR',
                     'get_last_retinal_imageG',
                     'get_last_retinal_imageB',
                     'get_last_optical_image',
                     'get_last_environment_map',
                     'save_last_environment_map',
                     'get_values',
                     'get_optics',
                     'full_spectrum',
                     ]:
            # 'subclassing' without subclassing
            setattr(self,attr, getattr(self.cvs,attr))

            #self.receptor_dirs = precomputed.receptor_dirs
            #self.emd_edges =
            self.emd_dirs_unrotated_quats = []
            for (vi1,vi2) in precomputed.edges:
                v1 = precomputed.receptor_dirs[vi1]
                v2 = precomputed.receptor_dirs[vi2]
                v = (v1+v2)*0.5 # average
                v = v.normalize() # make unit length
                self.emd_dirs_unrotated_quats.append( cgtypes.quat(0,v.x,v.y,v.z))

        if self.plot_realtime:
            if self.rp is None:
                self.rp = RapidPlotter(optics=self.optics)
                
            # strictly optional - realtime plotting stuff
            minx = numpy.inf
            maxx = -numpy.inf
            miny = numpy.inf
            maxy = -numpy.inf

            for num,eye_name in enumerate(self.rp.get_eye_names()):
                fans = self.rp.get_tri_fans(eye_name)
                x=[];y=[];f=[]
                for xs_ys in fans:
                    if xs_ys is None:
                        f.append(0)
                        continue
                    xs,ys=xs_ys
                    x.extend( xs )
                    y.extend( ys )
                    f.append( len(xs) )
                x = numpy.array(x)
                y = numpy.array(y)
                minx = min(minx,x.min())
                maxx = max(maxx,x.max())
                miny = min(miny,y.min())
                maxy = max(maxy,y.max())
                self.sim.set_eyemap_geometry(num, x, y, f)
            magx = maxx-minx
            if self.get_optics() == 'buchner71':
                if self.rp.flip_lon():
                    self.sim.set_eyemap_projection(0, maxx, minx-0.1*magx, miny, maxy)
                    self.sim.set_eyemap_projection(1, maxx+0.1*magx, minx, miny, maxy)
                else:
                    self.sim.set_eyemap_projection(0, minx-0.1*magx, maxx, miny, maxy)
                    self.sim.set_eyemap_projection(1, minx, maxx+0.1*magx, miny, maxy)
            elif self.get_optics() == 'synthetic':
                self.sim.set_eyemap_projection(0, minx-1*magx, maxx, miny, maxy)
                self.sim.set_eyemap_projection(1, minx, maxx+1*magx, miny, maxy)


    def get_root_node(self,*args,**kw):
        return self.sim.get_root_node(*args,**kw)

    def step(self, pos_vec3, ori_quat):
        if type(pos_vec3) != cgtypes.vec3:
            pos_vec3 = cgtypes.vec3( *pos_vec3 )

        if type(ori_quat) != cgtypes.quat:
            ori_quat = cgtypes.quat( *ori_quat )

        # calculate photoreceptors, EMDs, etc...
        self.cvs.step( self.sim.get_flyview, pos_vec3, ori_quat )

        self.last_pos_vec3 = pos_vec3
        self.last_ori_quat = ori_quat

        if self.plot_realtime:
            if self.full_spectrum:
                R = self.get_last_retinal_imageR()
                G = self.get_last_retinal_imageG()
                B = self.get_last_retinal_imageB()
            else:
                R = self.get_last_retinal_imageG()
                G = self.get_last_retinal_imageG()
                B = self.get_last_retinal_imageG()
            A = numpy.ones( B.shape, dtype=numpy.float32 )
            RGBA = numpy.vstack([R,G,B,A]).T
            RGBA[:,:3] = RGBA[:,:3]/255.0


            for num,eye_name in enumerate(self.rp.get_eye_names()):
                slicer = self.rp.get_slicer( eye_name )
                self.sim.set_eyemap_face_colors(num,RGBA[slicer,:])

    def get_known_node_parent(self,node):
        if node.getNumParents() != 1:
            raise NotImplementedError('only single parents currently supported!')
        parent = node.getParent(0)
        if parent in self.nodes:
            return parent
        else:
            return self.get_known_node_parent(parent)

    def get_last_retinal_velocities(self, vel_vec3, angular_vel, direction='ommatidia'):
        # NOTE: Both vel_vec3 and angular_vel should be in body frame

        x=self.last_pos_vec3 # eye origin
        q=self.last_ori_quat
        qi=q.inverse()

        vel_vecs = []
        mu_list = []
        dir_body_list = []
        dir_world_list = []

        if direction == 'ommatidia':
            dir_body_list = self.cvs.precomputed_optics_module.receptor_dirs
            for dir_body in dir_body_list:
                dir_world_quat = q*cgtypes.quat(0,dir_body.x, dir_body.y, dir_body.z)*qi
                dir_world_list.append(cgtypes.vec3(dir_world_quat.x,
                                                   dir_world_quat.y,
                                                   dir_world_quat.z))
        elif direction == 'emds':
            dir_body_quats = self.emd_dirs_unrotated_quats
            for dir_body_quat in dir_body_quats:
                dir_world_quat=q*dir_body_quat*qi # dir_body_quat.rotate(q)
                dir_world_list.append(cgtypes.vec3(dir_world_quat.x,
                                                   dir_world_quat.y,
                                                   dir_world_quat.z))
                dir_body_list.append(cgtypes.vec3(dir_body_quat.x,
                                                  dir_body_quat.y,
                                                  dir_body_quat.z))
        else:
            print 'ERROR: Directions need to be either ommatidia or EMDs.'
            quit()
            
        # Need a more efficient way to do this loop
        for n in range(len(dir_body_list)):
            dir_body = dir_body_list[n]
            dir_world = dir_world_list[n]

            vend = x + (dir_world*1e5) # XXX should figure out maximum length in OSG
            vstart = x + (dir_world*1e-5) # avoid intesecting with the origin
            
            rx, ry, rz, is_hit = self.sim.get_world_point(vstart, vend)
            
            if is_hit == 1: # is_hit is always 1 in the presence of skybox
                sigma = (cgtypes.vec3(rx,ry,rz) - x).length() # distance
                mu = 1.0/sigma
            else:
                mu = 0.0          # objects are at infinity

            # Use the formula in Sean Humbert's paper to calculate retinal velocity
            vr = - angular_vel.cross(dir_body) - mu * (vel_vec3 - dir_body * (dir_body*vel_vec3))

            # Convert vr into spherical coordinates
            # Equation: cf http://en.wikipedia.org/wiki/Vector_fields_in_cylindrical_and_spherical_coordinates
            sx = dir_body.x
            sy = dir_body.y
            sz = dir_body.z
            phi = numpy.math.atan2(sy,sx)
            theta = numpy.math.acos(sz)

            cphi = numpy.cos(phi)
            sphi = numpy.sin(phi)
            ctheta = numpy.cos(theta)
            stheta = numpy.sin(theta)

            R = cgtypes.mat3(stheta*cphi, stheta*sphi, ctheta, 
                             ctheta*cphi, ctheta*sphi, -stheta,
                             -sphi, cphi, 0)  # Row-major order
            vr = R*vr                         
            # vr[0]: r (should be zero), vr[1]: theta, vr[2]: phi
            vel_vecs.append([vr[1],vr[2]])
            mu_list.append(mu)
                        
        return vel_vecs, mu_list

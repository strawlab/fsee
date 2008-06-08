# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import cgkit.cgtypes as cgtypes # cgkit 2.x
import fsee
from fsee.CoreVisualSystem import CoreVisualSystem
from fsee.eye_geometry.projected_eye_coords import RapidPlotter

import os
if int(os.environ.get('FSEE_MULTIPROCESS','0')):
    import fsee.FlySimWrapFSOI as Simulation
else:
    import fsee.RealFSOI as Simulation

import fsee.eye_geometry.precomputed as precomputed

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

                 ):
        self.rp = RapidPlotter(optics=optics)
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

        if 1:
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

        if 1:
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

    def get_last_retinal_velocities(self, vel_vec3=None, angular_vel=None):
        # Retinal motion is in opposite direction from self motion, so
        # flip sign.
        angular_vel_from_rotation = -angular_vel

        x=self.last_pos_vec3 # eye origin
        tx=tuple(x)
        v3tx=osg.Vec3(*tx)

        q=self.last_ori_quat
        qi=q.inverse()

        root = self.sim.get_root_node()
        geode2known_node = {}

        vel_vecs = []

        for emd_dir_unrotated_quat in self.emd_dirs_unrotated_quats:
            # rotate emd_dirs
            emd_dir_quat=q*emd_dir_unrotated_quat*qi # q.rotate(emd_dir_unrotated_quat)
            emd_dir = cgtypes.vec3(emd_dir_quat.x,
                                   emd_dir_quat.y,
                                   emd_dir_quat.z)

            # Calculate translational component of retinal velocities

            # make unit length (should already be within floating
            # point error)
            emd_dir = emd_dir.normalize()

            y=x+(emd_dir*1e20) # XXX should figure out maximum length in OSG
            seg=osg.LineSegment()
            seg.set(v3tx,osg.Vec3(*tuple(y)))

            iv = osgUtil.IntersectVisitor()
            iv.addLineSegment(seg)
            root.accept(iv)

            if iv.hits():
                hitlist = iv.getHitList(seg)
                hit = hitlist.front()
                geode = hit.getGeode()
                # previous lookup results were cached, check there first
                known_node = geode2known_node.setdefault( geode, self.get_known_node_parent(geode) )
                #print 'found',self.nodes2names[known_node]

                hit_absolute_vel = self.nodes2vels[known_node]
                # XXX no account is taken of node's angular velocity (yet)

                if hit_absolute_vel is None:
                    # no relative translational motion, e.g. skybox
                    hit_relative_vel = cgtypes.vec3(0,0,0)
                    d0n = emd_dir
                    d1n = emd_dir
                else:
                    hit_relative_vel = hit_absolute_vel - vel_vec3

                    p = hit.getWorldIntersectPoint()
                    x0=cgtypes.vec3(p[0],p[1],p[2])
                    x1=x0+hit_relative_vel

                    if 0:
                        # slightly slower, but for clarity
                        d0=x0-x
                        d0n = d0.normalize()
                    else:
                        # slightly faster
                        d0n = emd_dir

                    d1=x1-x
                    d1n = d1.normalize()

            else:
                # this happens if there is no skybox
                d0n = emd_dir
                d1n = emd_dir

            # Now translational components are in d1n, rotate with angular velocity

            # d0n : EMD direction
            # d1n : object direction after object & observers translational velocities accounted for

            # projection of linear translation velocity onto sphere
            angular_vel_from_translation = d0n.angle(d1n) * d0n.cross(d1n)

            total_angular_vel = angular_vel_from_rotation + angular_vel_from_translation

            if 0:
                print 'd0n',d0n
                print 'd1n',d1n
                print 'angular_vel_from_translation',angular_vel_from_translation
                print 'total_angular_vel',total_angular_vel
                print

            vel_vecs.append( total_angular_vel )

        return vel_vecs

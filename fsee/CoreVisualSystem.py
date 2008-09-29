# Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
from __future__ import division
import fsee
import EMDSim

import cgtypes # cgkit 1.x
import Image

import numpy
import scipy
import math
from math import pi

import fsee.eye_geometry.switcher

##def get_orientation_quat(eye,center,up):
##    """find quaternion that rotates default OSG orientation to view direction"""
##    # XXX this doesn't appear to work yet...

##    # default OSG orientation = (ahead = -z, up = +y, right = +x)
##    print
##    print 'eye',eye
##    print 'center',center
##    print 'up',up
##    eye=cgtypes.vec3(eye)
##    center=cgtypes.vec3(center)
##    up=cgtypes.vec3(up)

####    viewdir = (center-eye).normalize()
####    upnorm = up.normalize()
####    print 'viewdir',viewdir
####    print 'upnorm',upnorm

####    if viewdir==cgtypes.vec3(0,0,-1) and upnorm == cgtypes.vec3(1,0,0):
####        # This is a hack/workaround because fromMat(m) will fail with
####        # zero division error in this case.
####        # looking at negz
####        q3 = cgtypes.quat(math.sqrt(0.5),0,0,-math.sqrt(0.5))
####    elif viewdir==cgtypes.vec3(0,0,1) and upnorm == cgtypes.vec3(-1,0,0):
####        # This is a hack/workaround because fromMat(m) will fail with
####        # zero division error in this case.
####        # looking at posz
####        q3 = cgtypes.quat(0,-math.sqrt(0.5),math.sqrt(0.5),0)
####    elif viewdir==cgtypes.vec3(0,1,0) and upnorm == cgtypes.vec3(0,0,1):
####        # This is a hack/workaround because fromMat(m) will fail with
####        # zero division error in this case.
####        # looking at posy
####        q3 = cgtypes.quat(math.sqrt(0.5),math.sqrt(0.5),0,0)
####    else:
##    if 1:

##        test_dir = cgtypes.vec4(1,0,0,1) # x,y,z


##        if 0:
##            viewdir = (center-eye).normalize()
##            print 'viewdir',viewdir
##            m = cgtypes.mat4().lookAt((0,0,0),viewdir,up)
##        elif 1:
##            viewdir = (center-eye).normalize()
##            up = up.normalize()

##            right = viewdir.cross(up).normalize()
##            #right = up.cross(viewdir).normalize()
##            print 'viewdir',viewdir
##            print 'right',right
##            print 'up',up
##            col1 = numpy.array([right.x, right.y, right.z, 0])
##            col2 = numpy.array([up.x, up.y, up.z, 0])
##            col3 = numpy.array([viewdir.x, viewdir.y, viewdir.z, 0])
##            col4 = numpy.array([0,0,0,1.0])
##            nm = numpy.c_[col1,col2,col3,col4].T
##            m = cgtypes.mat4( numpy.ravel(nm))
##        else:
##            m = cgtypes.mat4().lookAt(eye,center,up)
##        print 'm',m
##        print 'm*test_dir',m*test_dir
##        print 'test_dir*m',test_dir*m
##        q = cgtypes.quat().fromMat(m)
##        print 'q',q
##        m2 = q.toMat4()
##        print 'm2',m2
##        #q2 = cgtypes.quat().fromAngleAxis(pi/2,(1,0,0))
##        q2 = cgtypes.quat().fromAngleAxis(pi,(0,0,1))
##        print 'q2',q2
##        q3 = q2*q
##        print 'q3',q3
##        q3=q
##    return q3

##def test_orientation():
##    #                             center      up
##    viewdirs_lookat = {'posx':( ( 1, 0, 0), (0,0,1) ),
##                       'negx':( (-1, 0, 0), (0,0,1) ),
##                       'posy':( ( 0, 1, 0), (0,0,1) ),
##                       'negy':( ( 0,-1, 0), (0,0,1) ),
##                       'posz':( ( 0, 0, 1), (-1,0,0) ),
##                       'negz':( ( 0, 0,-1), (1,0,0) ),
##                       }

##    viewdirs_quat = {'posx':(cgtypes.quat().fromAngleAxis(-pi/2,(0,1,0))*
##                             cgtypes.quat().fromAngleAxis(-pi/2,(0,0,1))),
##                     'negx':(cgtypes.quat().fromAngleAxis(pi/2,(0,1,0))*
##                             cgtypes.quat().fromAngleAxis(pi/2,(0,0,1))
##                             ),
##                     'posy':cgtypes.quat().fromAngleAxis(pi/2,(1,0,0)),
##                     'negy':(cgtypes.quat().fromAngleAxis(-pi/2,(1,0,0))*
##                             cgtypes.quat().fromAngleAxis(pi,(0,0,1))),
##                     'posz':(cgtypes.quat().fromAngleAxis(pi/2,(0,0,1))*
##                             cgtypes.quat().fromAngleAxis(pi,(0,1,0))
##                             ),
##                     'negz':cgtypes.quat().fromAngleAxis(-pi/2,(0,0,1)),
##                     }
##    viewdirs = viewdirs_quat.keys()
##    viewdirs.sort()
##    for viewdir in viewdirs:
##        center, up = viewdirs_lookat[viewdir]
##        eye = (0,0,0)
##        quat = viewdirs_quat[viewdir]
##        try:
##            q2 = get_orientation_quat(eye,center,up)
##        except ZeroDivisionError:
##            print viewdir,'---> ',quat, 'err'
##        else:
##            print viewdir,'---> ',quat, q2

##def fastmul( cscmatrix, x ):
##    import warnings
##    warnings.warn("XXX Check that sparse matrix multiplication isn't dog-slow!")
##    return cscmatrix * x
####     func = getattr(scipy.sparse.sparsetools,cscmatrix.ftype+'cscmux')
####     y = func(cscmatrix.data, cscmatrix.rowind, cscmatrix.indptr, x, cscmatrix.shape[0])
####     return y

class CoreVisualSystem:
    def __init__(self,
                 hz=200.0,
                 earlyvis_ba = None, # early vision filter (can be None)
                 early_contrast_saturation_params=None,
                 emd_lp_ba = None, # EMD arm 2
                 emd_hp_ba = None, # EMD arm 1 (can be None)
                 subtraction_imbalance = 1.0,
                 debug = False,
                 full_spectrum=False,
                 cubemap_face_xres=64,
                 cubemap_face_yres=64,
                 optics=None,
                 do_luminance_adaptation=None,
                 ):
        precomputed = fsee.eye_geometry.switcher.get_module_for_optics(optics=optics)

        self.optics = optics
        self.precomputed_optics_module = precomputed

        self.cubemap_face_xres=cubemap_face_xres
        self.cubemap_face_yres=cubemap_face_yres

        self.full_spectrum=full_spectrum # do R and B channels (G always)


        self.cube_order = precomputed.cube_order

        # convert fsee body coordinates (ahead = +x, up = +z, right = -y)
        #   to OSG eye coordinates (ahead = -z, up = +y, right = +x)
        self.viewdirs = {'posx':(cgtypes.quat().fromAngleAxis(-pi/2,(0,1,0))*
                                 cgtypes.quat().fromAngleAxis(-pi/2,(0,0,1))),
                         'negx':(cgtypes.quat().fromAngleAxis(pi/2,(0,1,0))*
                                 cgtypes.quat().fromAngleAxis(pi/2,(0,0,1))
                                 ),
                         'posy':cgtypes.quat().fromAngleAxis(pi/2,(1,0,0)),
                         'negy':(cgtypes.quat().fromAngleAxis(-pi/2,(1,0,0))*
                                 cgtypes.quat().fromAngleAxis(pi,(0,0,1))),
                         'posz':(cgtypes.quat().fromAngleAxis(pi/2,(0,0,1))*
                                 cgtypes.quat().fromAngleAxis(pi,(0,1,0))
                                 ),
                         'negz':cgtypes.quat().fromAngleAxis(-pi/2,(0,0,1)),
                         }

        self.last_optical_images = {}
        for fn in self.viewdirs.keys():
            self.last_optical_images[fn] = None

        self.debug = debug
        if self.debug:
            self.responses_fd = open('01_retina.txt','wb')

        self.rweights = precomputed.receptor_weight_matrix_64.astype(numpy.float32)

        emd_edges = precomputed.edges

        n_receptors = self.rweights.shape[0]

        self.emd_sim = EMDSim.EMDSim(
            emd_hp_ba = emd_hp_ba, # highpass tau (can be None)
            emd_lp_ba = emd_lp_ba, # delay filter of EMD
            earlyvis_ba = earlyvis_ba, # if None, set to Howard, 1984
            early_contrast_saturation_params=early_contrast_saturation_params,
            subtraction_imbalance = subtraction_imbalance, # 1.0 for perfect subtraction
            compute_typecode = numpy.float32,
            hz=hz,
            n_receptors=n_receptors,
            emd_edges=emd_edges,
            do_luminance_adaptation=do_luminance_adaptation,
            )

        self.last_environment_map = None

    def get_optics(self):
        return self.optics

    def step(self, im_getter, pos_vec3, ori_quat ):
        self.last_environment_map = None # clear cached image if it exists...

        if type(pos_vec3) != cgtypes.vec3:
            pos_vec3 = cgtypes.vec3( *pos_vec3 )

        if type(ori_quat) != cgtypes.quat:
            ori_quat = cgtypes.quat( *ori_quat )

        # get direction of each face of the environment map cube as quat
        for fn, viewdir in self.viewdirs.iteritems():
            face_dir_quat = ori_quat*viewdir
            im=im_getter( pos_vec3, face_dir_quat )
            self.last_optical_images[fn] = im

        whole_env_mapG = numpy.concatenate( [numpy.ravel(self.last_optical_images[fn][:,:,1])
                                             for fn in self.cube_order], axis=0 )
        self.responsesG = self.rweights*whole_env_mapG # sparse matrix times vector = vector

        if self.full_spectrum:
            whole_env_mapR = numpy.concatenate( [numpy.ravel(self.last_optical_images[fn][:,:,0])
                                                 for fn in self.cube_order], axis=0 )
            self.responsesR = self.rweights * whole_env_mapR
            whole_env_mapB = numpy.concatenate( [numpy.ravel(self.last_optical_images[fn][:,:,2])
                                                 for fn in self.cube_order], axis=0 )
            self.responsesB = self.rweights * whole_env_mapB

        self.emd_sim.step( self.responsesG )

    def get_last_emd_outputs( self ):
        return self.emd_sim.emd_outputs[:,0] # original shape: (N,1)

    def get_values(self, *args, **kw ):
        return self.emd_sim.get_values(*args,**kw)

    def get_last_retinal_imageR(self):
        if not self.full_spectrum:
            raise ValueError('only compute G channel if not full spectrum')
        return self.responsesR

    def get_last_retinal_imageG(self):
        return self.responsesG

    def get_last_retinal_imageB(self):
        if not self.full_spectrum:
            raise ValueError('only compute G channel if not full spectrum')
        return self.responsesB

    def get_last_optical_image(self,facename):
        return self.last_optical_images[facename]

    def get_last_environment_map(self):
        if self.last_environment_map is None:
            faceh = self.cubemap_face_yres
            facew = self.cubemap_face_xres

            nchan = self.last_optical_images['posy'].shape[2]
            imnx = numpy.empty( (faceh*3,facew*4,nchan), dtype = numpy.uint8)
            if nchan == 4:
                # make background transparent (alpha channel = 0)
                initvalue = 0
            else:
                # make background white
                initvalue = 255
            imnx.fill(initvalue)

            yi0 = faceh
            yi1 = 2*faceh

            xi0 = 0
            xi1 = facew
            imnx[yi0:yi1,xi0:xi1,:] = self.last_optical_images['posy']

            xi0 = 2*facew
            xi1 = 3*facew
            imnx[yi0:yi1,xi0:xi1,:] = self.last_optical_images['negy']

            xi0 = 3*facew
            xi1 = 4*facew
            imnx[yi0:yi1,xi0:xi1,:] = self.last_optical_images['negx']

            xi0 = facew
            xi1 = 2*facew
            imnx[yi0:yi1,xi0:xi1,:] = self.last_optical_images['posx']

            yi0 = 0
            yi1 = faceh
            imnx[yi0:yi1,xi0:xi1,:] = self.last_optical_images['negz']

            yi0 = 2*faceh
            yi1 = 3*faceh
            imnx[yi0:yi1,xi0:xi1,:] = self.last_optical_images['posz']

            if nchan == 3:
                im = Image.fromstring('RGB',(imnx.shape[1],imnx.shape[0]),imnx.tostring())
            elif nchan == 4:
                im = Image.fromstring('RGBA',(imnx.shape[1],imnx.shape[0]),imnx.tostring())
            im = im.transpose( Image.FLIP_TOP_BOTTOM )
            self.last_environment_map = im
        return self.last_environment_map

    def save_last_environment_map(self,fname):
        im = self.get_last_environment_map()
        im.save(fname)

    def save_cubemap(self, center_x, center_y, center_z):
        # I don't think this works because it requires a 2nd OpenGL context.
        raise RuntimeError("Disabled -- probably doesn't work. Modify the source and check.")
        #self._flysim.save_cubemap( center_x, center_y, center_z)

##if __name__=='__main__':
##    test_orientation()


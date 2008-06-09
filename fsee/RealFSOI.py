# Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import numpy

import fsee
import os, sys
import pkg_resources

import ctypes
assert ctypes.__version__ >= '1.0.1'

if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
    backend_fname = 'libfsoi_ng.so'
elif sys.platform.startswith('win'):
    backend_fname = 'fsoi_ng.dll'
else:
    raise ValueError("unknown platform '%s'"%sys.platform)

backend_fullpath = pkg_resources.resource_filename(__name__,backend_fname)

############################
# import shared library

if sys.platform.startswith('linux'):
    c_fsoi_ng = ctypes.cdll.LoadLibrary(backend_fullpath)
elif sys.platform.startswith('win'):
    c_fsoi_ng = ctypes.CDLL(backend_fullpath)
elif sys.platform.startswith('darwin'):
    c_fsoi_ng = ctypes.CDLL(backend_fullpath)
else:
    raise ValueError("unknown platform '%s'"%sys.platform)

#############################################
# define function signatures and data types

class FsoiObj(ctypes.Structure):
    _fields_ = [('the_cpp_obj',ctypes.c_void_p),]

c_fsoi_ng.fsoi_ng_init.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_init.argtypes = []

c_fsoi_ng.fsoi_ng_shutdown.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_shutdown.argtypes = []

c_fsoi_ng.fsoi_ng_new.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_new.argtypes = [ctypes.POINTER(ctypes.POINTER(FsoiObj)),
                                  ctypes.c_char_p, # filename
                                  ctypes.c_double, # scale
                                  ctypes.c_char_p, # skybox_basename
                                  ctypes.c_double, # im_xang
                                  ctypes.c_double, #
                                  ctypes.c_double, # near
                                  ctypes.c_double, #
                                  ctypes.c_int, # width
                                  ctypes.c_int, # height
                                  ctypes.c_char_p, # render_implementation
                                  ]

c_fsoi_ng.fsoi_ng_delete.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_delete.argtypes = [ctypes.POINTER(FsoiObj)]

c_fsoi_ng.fsoi_ng_run.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_run.argtypes = [ctypes.POINTER(FsoiObj)]

c_fsoi_ng.fsoi_ng_render_frame.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_render_frame.argtypes = [ctypes.POINTER(FsoiObj),
                                           ctypes.c_void_p, # image_data_ptr
                                           ctypes.POINTER(ctypes.c_int), # width
                                           ctypes.POINTER(ctypes.c_int), # height
                                           ctypes.POINTER(ctypes.c_int), # bytes per pixel
                                           ]

c_fsoi_ng.fsoi_ng_render_frame_copy.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_render_frame_copy.argtypes = [ctypes.POINTER(FsoiObj),
                                                ctypes.c_void_p, # image_data
                                                ctypes.c_int, # width
                                                ctypes.c_int, # height
                                                ctypes.c_int, # bytes per pixel
                                                ]

c_fsoi_ng.fsoi_ng_set_pos_ori.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_set_pos_ori.argtypes = [ctypes.POINTER(FsoiObj),
                                          ctypes.c_double,
                                          ctypes.c_double,
                                          ctypes.c_double,

                                          ctypes.c_double,
                                          ctypes.c_double,
                                          ctypes.c_double,
                                          ctypes.c_double,
                                          ]

c_fsoi_ng.fsoi_ng_set_eyemap_geometry.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_set_eyemap_geometry.argtypes = [ctypes.POINTER(FsoiObj),
                                                  ctypes.c_int,
                                                  ctypes.c_void_p,
                                                  ctypes.c_int,
                                                  ctypes.c_void_p,
                                                  ctypes.c_int]

c_fsoi_ng.fsoi_ng_set_eyemap_projection.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_set_eyemap_projection.argtypes = [ctypes.POINTER(FsoiObj),
                                                    ctypes.c_int,
                                                    ctypes.c_float,
                                                    ctypes.c_float,
                                                    ctypes.c_float,
                                                    ctypes.c_float]
c_fsoi_ng.fsoi_ng_set_eyemap_face_colors.restype = ctypes.c_int
c_fsoi_ng.fsoi_ng_set_eyemap_face_colors.argtypes = [ctypes.POINTER(FsoiObj),
                                                     ctypes.c_int,
                                                     ctypes.c_void_p,
                                                     ctypes.c_int]
############################

class FSOIError(Exception):
    pass

def CHK(m):
    if m!=0:
        raise FSOIError('error: %d'%(m,))

CHK(c_fsoi_ng.fsoi_ng_init())

class Simulation:
    def __init__(self,
                 model_path=None,
                 scale=1.0,
                 skybox_basename="Images/brightday1/",
                 xres=64,
                 yres=64,
                 xang=90.0,
                 yang=90.0,
                 zmin=0.001,
                 zmax=1e10):
        self.fsoi = ctypes.pointer(FsoiObj())

        CHK(c_fsoi_ng.fsoi_ng_new(ctypes.byref(self.fsoi),
                                  model_path,
                                  scale,
                                  skybox_basename,
                                  xang,
                                  yang,
                                  zmin,
                                  zmax,
                                  xres,
                                  yres,
                                  "fb"))


    def __del__(self):
        if hasattr(self,'fsoi'):
            CHK(c_fsoi_ng.fsoi_ng_delete(self.fsoi))
            del self.fsoi

    def get_flyview(self,pos_vec3,ori_quat):
        """returns color image according to position and orientation"""

        CHK(c_fsoi_ng.fsoi_ng_set_pos_ori(self.fsoi,
                                          pos_vec3.x,
                                          pos_vec3.y,
                                          pos_vec3.z,
                                          ori_quat.w,
                                          ori_quat.x,
                                          ori_quat.y,
                                          ori_quat.z,
                                          ))
        width = ctypes.c_int()
        height = ctypes.c_int()

        CHK(c_fsoi_ng.fsoi_ng_get_width_height(self.fsoi,ctypes.byref(width),ctypes.byref(height)))

        buf = numpy.empty( (height.value,width.value,4), dtype=numpy.uint8) # RGBA uint8

        if isinstance(buf.ctypes.data,ctypes.c_void_p):
            data_ptr = buf.ctypes.data
        else:
            data_ptr = ctypes.c_void_p(buf.ctypes.data)

        # memcopy
        bytes_per_pixel = 4
        CHK(c_fsoi_ng.fsoi_ng_render_frame_copy(self.fsoi,data_ptr,width,height,bytes_per_pixel))
        return buf

    def set_eyemap_geometry(self, num, face_xs, face_ys, face_fans):
        face_xys = numpy.ravel( numpy.array( [face_xs, face_ys], dtype=numpy.float32 ).T )
        face_fans = numpy.array( face_fans, dtype=numpy.uint8 )
        CHK(c_fsoi_ng.fsoi_ng_set_eyemap_geometry(self.fsoi,num,
                                                  face_xys.ctypes.data,len(face_xs),
                                                  face_fans.ctypes.data,len(face_fans)))

    def set_eyemap_projection(self, num, x1, x2, y1, y2 ):
        CHK(c_fsoi_ng.fsoi_ng_set_eyemap_projection(self.fsoi,num,x1, x2, y1, y2 ))

    def set_eyemap_face_colors(self, num, RGBA_array ):
        assert RGBA_array.shape[1] == 4
        assert RGBA_array.dtype == numpy.float32
        data = numpy.ravel(RGBA_array)
        CHK(c_fsoi_ng.fsoi_ng_set_eyemap_face_colors(self.fsoi,num,data.ctypes.data,len(RGBA_array)))

##    def render_image(self, pos_vec3, ori_quat,
##                     my_xres, my_yres, my_xang, my_yang,
##                     near, far):
##        """returns my_xres x my_yres color image"""

##        dims = (my_yres,my_xres,3)
##        im = numpy.empty(dims,dtype=numpy.uint8)

##        pos_x, pos_y, pos_z = pos_vec3.x, pos_vec3.y, pos_vec3.z
##        ori_x, ori_y, ori_z, ori_w = ori_quat.x, ori_quat.y, ori_quat.z, ori_quat.w

##        self.fsoi.render_image(
##            pos_x, pos_y, pos_z,
##            ori_x, ori_y, ori_z, ori_w,
##            my_xres, my_yres, my_xang, my_yang,
##            im,near,far)
##        return im

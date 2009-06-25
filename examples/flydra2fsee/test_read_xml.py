from __future__ import division
from __future__ import with_statement
import pkg_resources
if 1:
    # deal with old files, forcing to numpy
    import tables.flavor
    tables.flavor.restrict_flavors(keep=['numpy'])
import sets, os, sys, math, time

import numpy
import tables as PT
from optparse import OptionParser
import flydra.a2.xml_stimulus as xml_stimulus
import flydra.a2.xml_stimulus_osg as xml_stimulus_osg
import flydra.a2.core_analysis as core_analysis
import flydra.a2.analysis_options as analysis_options
import flydra.analysis.result_utils as result_utils
import flydra.a2.flypos
import fsee
import fsee.Observer
import fsee.plot_utils
import pylab

import cgtypes

class PathMaker:
    def __init__(self,
                 pos_0=cgtypes.vec3(0,0,0),
                 ori_0=cgtypes.quat(1,0,0,0),
                 vel_0=cgtypes.vec3(0,0,0),
                 ):
        self.pos_0 = pos_0
        self.ori_0 = ori_0
        self.vel_0 = vel_0
        self.reset()
    def reset(self):
        self.cur_pos = self.pos_0
        self.cur_ori = self.ori_0
        self.vel = self.vel_0
        self.angular_vel = cgtypes.vec3(0,0,0) # fixed for now
        self.last_t = None
    def step(self,t):
        if self.last_t is not None:
            delta_t = t-self.last_t
        else:
            delta_t = 0
        self.last_t = t
        self.cur_pos += self.vel*delta_t

        return self.cur_pos, self.cur_ori, self.vel, self.angular_vel

def main():
    # Generate OSG model from the XML file
    stim_xml_filename = 'nopost.xml'
    stim_xml = xml_stimulus.xml_stimulus_from_filename(stim_xml_filename)
    stim_xml_osg = xml_stimulus_osg.StimulusWithOSG( stim_xml.get_root() )

    # Simulation parameters
    hz = 60.0 # fps
    dt = 1.0/hz

    # Use to generate pose and velocities
    path_maker = PathMaker(vel_0=cgtypes.vec3(0.100,0,0),
                           pos_0=cgtypes.vec3(0.000,0.000,0.150))


    with stim_xml_osg.OSG_model_path() as osg_model_path:
        vision = fsee.Observer.Observer(model_path=osg_model_path,
                                        # scale=1000.0, # from meters to mm
                                        hz=hz,
                                        skybox_basename=None,
                                        full_spectrum=True,
                                        optics='buchner71',
                                        do_luminance_adaptation=False,
                                        )
        t = -dt 
        count = 0
        y_old = None
        cur_pos = None
        cur_ori = None
 
        while count < 1000:
            t += dt
            count += 1

            cur_pos, cur_ori, vel, angular_vel = path_maker.step(t)
            print cur_pos

            vision.step(cur_pos,cur_ori)

            R=vision.get_last_retinal_imageR()
            G=vision.get_last_retinal_imageG()
            B=vision.get_last_retinal_imageB()
            y = (R+G+B)/3.0/255
            
            if y_old != None:
                ydot = (y - y_old) / dt
            else:
                y_old = y
                    
            # Get Q_dot and mu from body velocities
            # WARNING: Getting Q_dot and mu can be very slow
            qdot, mu = vision.get_last_retinal_velocities(vel, angular_vel)

if __name__=='__main__':
    main()


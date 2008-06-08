# Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import cgkit.cgtypes as cgtypes # cgkit 2
import math
import numpy
import sets

def project_angular_velocity_to_emds(angular_vels, receptor_dirs, edges):
    # angular_vels comes from same data as edges
    assert len(angular_vels) == len( edges )

    emd_outputs = []
    zero_vec = cgtypes.vec3(0,0,0)
    for (vi0,vi1),angular_vel in zip(edges,angular_vels):
        if angular_vel == zero_vec:
            emd_outputs.append( 0.0 )
            continue
        # find center of EMD correlation pair and its preferred direction
        v0 = receptor_dirs[vi0]
        v1 = receptor_dirs[vi1]
        emd_lpd_dir = (v0-v1).normalize() # local preferred direction
        emd_pos = ((v0+v1)*0.5).normalize() # EMD center

        # project angular velocity onto EMD baseline
        tangential_vel = angular_vel.cross( emd_pos )
        theta = emd_lpd_dir.angle( tangential_vel )
        projected_mag = abs(tangential_vel)*math.cos( theta )
        emd_outputs.append( projected_mag )
    return emd_outputs


def project_emds_to_angular_velocity(emd_signals,receptor_dirs,edges):
    # emd_signals comes from same data as edges
    assert len(emd_signals) == len( edges )
    angular_vels = []
    for (vi0,vi1),emd_signal in zip(edges,emd_signals):
        # find center of EMD correlation pair and its preferred direction
        v0 = receptor_dirs[vi0]
        v1 = receptor_dirs[vi1]
        emd_lpd_dir = (v0-v1).normalize() # local preferred direction
        emd_pos = ((v0+v1)*0.5).normalize() # EMD center

        # project EMD baseline by signal
        emd_resp = float(emd_signal)*emd_lpd_dir
        angular_vel = emd_pos.cross( emd_resp )
        angular_vels.append(angular_vel)
    return angular_vels

def get_emd_center_directions( edges, receptor_dirs ):
    emd_center_dirs = []
    for (vi0,vi1) in edges:
        # find center of EMD correlation pair and its preferred direction
        v0 = receptor_dirs[vi0]
        v1 = receptor_dirs[vi1]
        emd_lpd_dir = (v0-v1).normalize() # local preferred direction
        emd_pos = ((v0+v1)*0.5).normalize() # EMD center
        emd_center_dirs.append( emd_pos )
    return emd_center_dirs


def get_mean_interommatidial_distance( receptor_dirs, triangles ):
    """returns values in radians"""
    # this is not efficient...
    mean_thetas = []
    for iv,v in enumerate(receptor_dirs):
        neighbors = sets.Set()
        for tri in triangles:
            if iv in tri:
                for it in tri:
                    neighbors.add(it)
        neighbors = list(neighbors)
        neighbors.remove( iv )
        neighbor_dirs = [ receptor_dirs[int(n)] for n in neighbors ]
        cos_theta_neighbors = [numpy.dot(n,v) for n in neighbor_dirs]
        theta_neighbors = [numpy.arccos( c ) for c in cos_theta_neighbors]
        mean_theta = numpy.mean(theta_neighbors)
        mean_thetas.append(mean_theta)
    return mean_thetas

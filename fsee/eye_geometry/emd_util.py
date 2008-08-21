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

def find_edges( tris ):
    def sort_pair( ab ):
        if ab[0] <= ab[1]:
            return ab
        else:
            return (ab[1], ab[0])
    def is_in_sorted_pairs( edge, edges ):
        e0 = edge[0]
        e1 = edge[1]
        # we know edge[0] is less than edge[1], and pairs in edges are similarly sorted
        for etest in edges:
            if e0 == etest[0]:
                if e1 == etest[1]:
                    return True
        return False

    edges = []
    for tri in tris:
        tri_edges = [sort_pair((tri[0], tri[1])),
                     sort_pair((tri[1], tri[2])),
                     sort_pair((tri[2], tri[0]))]
        for tri_edge in tri_edges:
            if not is_in_sorted_pairs(tri_edge, edges):
                edges.append( tri_edge )
    return edges

def pseudo_voronoi(verts,tris_orig):
    if hasattr(tris_orig,'shape'):
        # convert from numpy to list of tuples for comparison
        tris = [ tuple(tri) for tri in tris_orig ]
    else:
        tris = tris_orig
    faces=[]
    for vert_idx,vert in enumerate(verts):
        # find my triangles
        my_tris = []
        for tri in tris:
            if vert_idx in tri:
                my_tris.append( tri )

        # now walk around neighboring triangles
        ordered_facet = []
        ordered_tris = []
        # start walk at first triangle in list
        prev_tri = my_tris[0]
        ordered_tris.append( prev_tri )
        prev_vert = my_tris[0][0]
        if prev_vert == vert_idx:
            prev_vert = my_tris[0][1]
        ordered_facet.append( prev_vert )
        tri_idx = 1
        while len(ordered_facet) < len( my_tris ):
            tri = my_tris[tri_idx]
            tri_idx = (tri_idx+1)%len(my_tris) # wrap to beginning
            if tri == prev_tri:
                continue
            if prev_vert not in tri: # must walk
                continue

            for nvidx in tri:
                if nvidx == vert_idx or nvidx == prev_vert:
                    continue
                ordered_facet.append( nvidx )
                prev_vert = nvidx
                prev_tri = tri
                ordered_tris.append( prev_tri )
                break

        #vverts = [ (vert + verts[ofvi])*0.5 for ofvi in ordered_facet ]
        vverts = [ (verts[t[0]]+verts[t[1]]+verts[t[2]])*(1.0/3.0) for t in ordered_tris ]
        faces.append( vverts )
    return faces

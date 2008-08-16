#!/usr/bin/env python
# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
from __future__ import division

import math, sys, sets, os
import cgkit # cgkit 2.x
import cgkit.cgtypes as cgtypes
import numpy
import scipy
import scipy.sparse
import scipy.io
from drosophila_eye_map.util import cube_order, make_repr_able, save_as_python, \
     make_receptor_sensitivities, flatten_cubemap
from emd_util import find_edges, pseudo_voronoi

# XXX we should make delta_phi depend on distance to nearest neighbors
# and thus be space-variant.

def make_receptor_dirs(n_subdivides=3):
    # subdivides icosahedron.
    # http://en.wikipedia.org/wiki/Icosahedron
    phi = (1+math.sqrt(5))/2

    verts = []

    def make_vert(*args):
        return cgtypes.vec3(*args)
        #return args

    for o in [-1,1]:
        for p in [-phi,phi]:
            verts.append( make_vert(0, o, p) )

    for o in [-1,1]:
        for p in [-phi,phi]:
            verts.append( make_vert(o, p, 0) )

    for o in [-1,1]:
        for p in [-phi,phi]:
            verts.append( make_vert(p, 0, o) )

    if 0:
        # This was piped to qhull "qhull i < receptors.qhull"
        print '3'
        print len(verts)
        for v in verts:
            print ' '.join(map(repr,v))

    #
    qres= """5 10 8
    10 3 1
    3 10 5
    0 6 9
    2 5 8
    0 2 8
    2 0 9
    4 0 8
    4 10 1
    10 4 8
    6 4 1
    4 6 0
    11 6 1
    3 11 1
    6 11 9
    2 7 5
    7 3 5
    7 2 9
    11 7 9
    7 11 3"""
    qres = map(int,qres.split())
    tris = []
    for i in range(len(qres)//3):
        tris.append( (qres[i*3], qres[i*3+1], qres[i*3+2]) )

    def subdivide( orig_verts, orig_tris ):

        def find_vert(testv4, new_verts, eps=1e-10):
            for vi,v in enumerate(new_verts):
                if abs(testv4-v) < eps:
                    return vi
            return None

        new_tris = []
        new_verts = orig_verts[:] # copy
        for orig_tri in orig_tris:

            v0i = orig_tri[0]
            v0 = orig_verts[v0i]

            v1i = orig_tri[1]
            v1 = orig_verts[v1i]

            v2i = orig_tri[2]
            v2 = orig_verts[v2i]

            testv3 = (v0+v1)*0.5
            v3i = find_vert(testv3, new_verts)
            if v3i is None:
                v3i = len(new_verts)
                v3 = testv3
                new_verts.append(v3)

            testv4 = (v1+v2)*0.5
            v4i = find_vert(testv4, new_verts)
            if v4i is None:
                v4i = len(new_verts)
                v4 = testv4
                new_verts.append(v4)

            testv5 = (v2+v0)*0.5
            v5i = find_vert(testv5, new_verts)
            if v5i is None:
                v5i = len(new_verts)
                v5 = testv5
                new_verts.append(v5)

            new_tris.append( (v0i, v3i, v5i) )
            new_tris.append( (v3i, v1i, v4i) )
            new_tris.append( (v5i, v4i, v2i) )
            new_tris.append( (v5i, v3i, v4i) )
        return new_verts, new_tris

    for i in range(n_subdivides):
        verts, tris = subdivide( verts, tris )

    verts = [ vert*(1.0/abs(vert)) for vert in verts ] # normalize

    return verts, tris

def sort_receptors_by_phi(receptor_dirs,nbins = 32):

    def xyz2lonlat(x,y,z):
        R2D = 180/math.pi
        proj = 'cyl'
        lat = math.asin(z)*R2D
        lon1 = math.atan2(y,x)*R2D
        if proj=='cyl':
            lon1 = -lon1
        #lon2 = (lon1+180.0+180.0)%360.0-180.0 # shift +X to 0 longitutde
        return lon1,lat

    bin_lons = numpy.arange(-180,180,(360.0/nbins))

    rdirs=receptor_dirs

    rdirs2 = [ xyz2lonlat( *rdir ) for rdir in rdirs ]
    lons, lats = zip(*rdirs2)

    weight_matrix = numpy.zeros( (nbins,len(rdirs)), dtype=numpy.float32 )

    count = 0
    for j, bin_lon_start in enumerate(bin_lons):
        bin_lon_end = bin_lon_start + 360.0/nbins
        this_row_weights = numpy.zeros( (len(rdirs),), dtype=numpy.float32 )

        for i,rdir in enumerate(rdirs):
            rdir_lon = lons[i]
            rdir_lat = lats[i]
            left_vert = rdir_lon
            if not (left_vert >= bin_lon_start and left_vert < bin_lon_end):
                # not in bin
                continue

##            if rdir_lat > 0: # above horizon
##                weight = math.cos( rdir_lat*math.pi/180.0 ) # weight by cos(latitude)
##                weight_matrix[j,i] = weight

##            if 1:
##                weight = math.cos( rdir_lat*math.pi/180.0 ) # weight by cos(latitude)
##                weight_matrix[j,i] = weight

            if 35 < rdir_lat < 60:
                #weight = math.cos( rdir_lat*math.pi/180.0 ) # weight by cos(latitude)
                weight = 1.0
                weight_matrix[j,i] = weight

##    for i in range(nbins):
##        #print weight_matrix[i,:]
##        sum_row = scipy.sum(weight_matrix[i,:])
##        print sum_row
##        #print
##        weight_matrix[i,:] = weight_matrix[i,:] / sum_row

    weight_matrix_sparse = scipy.sparse.csc_matrix(weight_matrix)
    return weight_matrix_sparse

if __name__ == '__main__':
    script_dir = os.path.abspath(os.path.split(__file__)[0])
    os.chdir(script_dir)

########################################################
#    SAVE INFO (from save_sparse_weights.py)
########################################################

    receptor_dirs, tris = make_receptor_dirs(n_subdivides=3) # n_subidivisions
    receptors_by_phi = sort_receptors_by_phi(receptor_dirs,nbins = 32)
    edges = find_edges( tris )
    verts = receptor_dirs

    rad2deg = 180/math.pi
    v0 = verts[0]
    a_degs = [v0.angle(v)*rad2deg for v in verts[1:]]
    a_degs.sort()
    delta_phi_deg = a_degs[0] # inter receptor angle, 6.848549293 when 3 subdivisions of icosahedron
    print 'delta_phi_deg',delta_phi_deg
    delta_phi = delta_phi_deg/rad2deg

    delta_rho = delta_phi * 1.1 # rough approximation. follows from caption of Fig. 18, Buchner, 1984 (in Ali)

    weight_maps_64 = make_receptor_sensitivities( receptor_dirs, delta_rho_q=delta_rho, res=64 )
    print 'weight_maps calculated'

    #####################################

    clip_thresh=1e-5
    floattype=numpy.float32

    weights = flatten_cubemap( weight_maps_64[0] ) # get first one to take size

    n_receptors = len(receptor_dirs)
    len_wm = len(weights)

    print 'allocating memory...'
    bigmat_64 = numpy.zeros( (n_receptors, len_wm), dtype=floattype )
    print 'done'

    print 'flattening, clipping, casting...'
    for i, weight_cubemap in enumerate(weight_maps_64):
        weights = flatten_cubemap( weight_cubemap )
        if clip_thresh is not None:
            weights = numpy.choose(weights<clip_thresh,(weights,0))
        bigmat_64[i,:] = weights.astype( bigmat_64.dtype )
    print 'done'

    print 'worst gain (should be unity)',min(numpy.sum( bigmat_64, axis=1))
    print 'filling spmat_64...'
    sys.stdout.flush()
    spmat_64 = scipy.sparse.csc_matrix(bigmat_64)
    print 'done'

    M,N = bigmat_64.shape
    print 'Compressed to %d of %d'%(len(spmat_64.data),M*N)

    faces = pseudo_voronoi(receptor_dirs,tris)

    ##################################################
    # Save matlab version

    fd = open('precomputed_synthetic.m','w')
    fd.write( 'receptor_dirs = [ ...')
    for rdir in receptor_dirs:
        fd.write( '\n    %s %s %s;'%( repr(rdir[0]), repr(rdir[1]), repr(rdir[2]) ) )
    fd.write( '];\n\n')

    fd.write( 'edges = [ ...')
    for e in edges:
        fd.write( '\n    %d %d;'%( e[0]+1, e[1]+1 )) # convert to 1-based indexing
    fd.write( '];\n\n')
    fd.close()
    ##################################################

    receptor_dir_slicer = {None:slice(0,len(receptor_dirs),1)}
    edge_slicer = {None:slice(0,len(edges),1)}
    #

    fd = open('precomputed_synthetic.py','wb')
    fd.write( '# Automatically generated by %s\n'%os.path.split(__name__)[-1])
    fd.write( 'import numpy\n')
    fd.write( 'import scipy\n')
    fd.write( 'import scipy.sparse\n')
    fd.write( 'import scipy.io\n')
    fd.write( 'import cgkit.cgtypes # cgkit 2\n')
    fd.write( 'import os\n')
    fd.write( 'datadir = os.path.split(__file__)[0]\n')
    fd.write( 'cube_order = %s\n'%repr(cube_order) )
    save_as_python(fd, receptor_dir_slicer, 'receptor_dir_slicer', fname_extra='_synthetic' )
    save_as_python(fd, edge_slicer, 'edge_slicer', fname_extra='_synthetic' )
    save_as_python(fd, spmat_64, 'receptor_weight_matrix_64', fname_extra='_synthetic' )
    save_as_python(fd, map(make_repr_able,receptor_dirs), 'receptor_dirs', fname_extra='_synthetic' )
    save_as_python(fd, tris, 'triangles')
    save_as_python(fd, edges, 'edges')
    save_as_python(fd, map(make_repr_able,faces), 'hex_faces')
    save_as_python(fd, receptors_by_phi, 'receptors_by_phi',fname_extra='_synthetic' )
    fd.write( '\n')
    fd.write( '\n')
    fd.write( '\n')
    extra = open('plot_receptors_vtk.py','r').read()
    fd.write( extra )
    fd.close()


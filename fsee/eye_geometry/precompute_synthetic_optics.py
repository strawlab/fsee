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

# XXX we should make delta_phi depend on distance to nearest neighbors
# and thus be space-variant.

cube_order = ['posx','negx','posy','negy','posz','negz']

class repr_vec3(cgtypes.vec3):
    def __repr__(self):
        return 'cgkit.cgtypes.vec3(%s, %s, %s)'%( repr(self.x),
                                                  repr(self.y),
                                                  repr(self.z) )

class repr_quat(cgtypes.quat):
    def __repr__(self):
        return 'cgkit.cgtypes.quat(%s, %s, %s, %s)'%( repr(self.w),
                                                      repr(self.x),
                                                      repr(self.y),
                                                      repr(self.z) )

def make_repr_able(x):
    if isinstance(x, cgkit.cgtypes.vec3):
        return repr_vec3(x)
    elif isinstance(x, cgkit._core.vec3):
        return repr_vec3(x)
    elif isinstance(x, cgkit.cgtypes.quat):
        return repr_quat(x)
    elif isinstance(x, cgkit._core.quat):
        return repr_quat(x)
    elif isinstance(x, list):
        # recurse into
        y = map( make_repr_able,x)
        return y
    else:
        return x

def test_repr():
    x = repr_vec3(1,2,3.0000001)
    ra = repr(x)
    x2 = eval(ra)
    assert x2.z == x.z

    y = [cgkit.cgtypes.vec3(1,2,3.0000001)]
    y2 = map(make_repr_able,y)
    assert y[0].z == y2[0].z

    x = repr_quat(0.1,1,2,3.0000001)
    ra = repr(x)
    x2 = eval(ra)
    assert x2.z == x.z

    y = [cgkit.cgtypes.quat(0.1,1,2,3.0000001)]
    y2 = map(make_repr_able,y)
    assert y[0].z == y2[0].z

    y3 = [y]
    y4 = map(make_repr_able,y3)
    assert y3[0][0].z == y4[0][0].z

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

def mag(vec):
    vec = numpy.asarray(vec)
    assert len(vec.shape)==1
    return math.sqrt(numpy.sum(vec**2.0))

def normalize(vec):
    denom = mag(vec)
    return numpy.asarray(vec)/denom

def make_receptor_sensitivities(all_d_q,delta_rho_q=None,res=64):
    """

    all_d_q are visual element directions as a 3-vector
    delta_rho_q (angular sensitivity) is in radians

    """
    if delta_rho_q is None:
        raise ValueError('must specify delta_rho_q (in radians)')

    if isinstance( delta_rho_q, float):
        all_delta_rho_qs = delta_rho_q*numpy.ones( (len(all_d_q),), dtype=numpy.float64)
    else:
        all_delta_rho_qs = numpy.asarray(delta_rho_q)
        if len(all_delta_rho_qs.shape) != 1:
            raise ValueError("delta_rho_q must be scalar or vector")
        if all_delta_rho_qs.shape[0] != len(all_d_q):
            raise ValueError("if delta_rho_q is a vector, "
                             "it must have the same number of "
                             "elements as receptors")

    def G_q(zeta,delta_rho_q):
        # gaussian
        # From Snyder (1979) as cited in Burton & Laughlin (2003)
        return numpy.exp( -4*math.log(2)*abs(zeta)**2 / delta_rho_q**2 )

    half_res = res//2
    vals = (numpy.arange(res)-half_res)/half_res

    weight_maps = []

    # setup vectors for initial face (posx)
    face_vecs = {}
    face_vecs['posx'] = []
    x = 1
    for z in vals:
        this_row_vecs = []
        for y in vals:
            on_cube_3d = (x,y,z)
            #print 'on_cube_3d   %5.2f %5.2f %5.2f'%on_cube_3d
            v3norm = normalize(on_cube_3d) # get direction of each pixel
            p_p = cgtypes.quat(0.0, v3norm[0], v3norm[1], v3norm[2])
            this_row_vecs.append(p_p)
        this_row_vecs.reverse()
        face_vecs['posx'].append( this_row_vecs )

    def rot_face( facedict, facename, rotq):
        facedict[facename] = []
        for row in facedict['posx']:
            this_row_vecs = []
            for col in row:
                this_row_vecs.append( rotq*col*rotq.inverse() )
            facedict[facename].append( this_row_vecs )

    rotq = cgtypes.quat()
    rotq = rotq.fromAngleAxis(math.pi/2.0,cgtypes.vec3(0,0,1))
    rot_face( face_vecs, 'posy', rotq)

    rotq = cgtypes.quat()
    rotq = rotq.fromAngleAxis(math.pi,cgtypes.vec3(0,0,1))
    rot_face( face_vecs, 'negx', rotq)

    rotq = cgtypes.quat()
    rotq = rotq.fromAngleAxis(-math.pi/2.0,cgtypes.vec3(0,0,1))
    rot_face( face_vecs, 'negy', rotq)

    rotq = cgtypes.quat()
    rotq = rotq.fromAngleAxis(math.pi/2.0,cgtypes.vec3(0,-1,0))
    rot_face( face_vecs, 'posz', rotq)

    rotq = cgtypes.quat()
    rotq = rotq.fromAngleAxis(math.pi/2.0,cgtypes.vec3(0,1,0))
    rot_face( face_vecs, 'negz', rotq)

    # convert from quat to vec3
    rfv = {}
    for key, rows in face_vecs.iteritems():
        rfv[key] = []
        for row in rows:
            this_row = [ cgtypes.vec3(col.x, col.y, col.z) for col in row ] # convert to vec3
            rfv[key].append( this_row )

    def get_weight_map(fn, rfv, d_q, delta_rho_q):
        angles = numpy.zeros( (vals.shape[0], vals.shape[0]), dtype=numpy.float64 )
        for i, row_vecs in enumerate(rfv[fn]):
            for j, ovec in enumerate(row_vecs):
                angles[i,j] = d_q.angle(ovec)
        wm = G_q(angles,delta_rho_q)
        return wm

    for dqi,(d_q,this_delta_rho_q) in enumerate(zip(all_d_q,all_delta_rho_qs)):
        print '%d of %d'%(dqi+1, len(all_d_q))
        weight_maps_d_q = {}
        ssf = 0.0

        for fn in cube_order:
            wm = get_weight_map(fn, rfv, d_q, this_delta_rho_q)
            weight_maps_d_q[fn] = wm
            ssf += numpy.sum( wm.flat )

        # normalize
        for mapname,wm in weight_maps_d_q.iteritems():
            weight_maps_d_q[mapname] = wm/ssf

        # save maps by receptor direction
        weight_maps.append( weight_maps_d_q )
    return weight_maps

def flatten_cubemap( cubemap ):
    rank1 = numpy.concatenate( [ numpy.ravel(cubemap[dir]) for dir in cube_order], axis=0 )
    return rank1

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

def get_code_for_var( name, fname_prefix, var):
    if (isinstance(var,numpy.ndarray) or
        scipy.sparse.issparse(var)):

        if 0:
            # save as Matrix Market file
            fname = fname_prefix + '.mtx'
            scipy.io.mmwrite( fname, var )

            result = '%s = scipy.io.mmread(os.path.join(datadir,"%s"))\n'%(name,fname)
        else:
            # save as compressed MATLAB .mat file
            fname = fname_prefix + '.mat'
            fd = open( fname, mode='wb' )
            savedict = {name:var}
            #scipy.io.savemat(fname, savedict, format='5' )
            scipy.io.savemat(fd, savedict)
            result = '%s = scipy.io.loadmat(open(os.path.join(datadir,"%s"),mode="rb"))\n'%(name,fname)
        return result

    if 1:
        ra = repr(var)
        # now check that conversion worked
        try:
            cmp = eval(ra)
        except Exception, err:
            import traceback
            print 'the following exception will trigger a RuntimeError("eval failed") call:'
            traceback.print_exc()
            raise RuntimeError("eval failed")
        else:
            if cmp==var:
                return '%s = '%(name,)+ra+'\n'
            else:
                raise RuntimeError("failed conversion for %s (type %s)"%(repr(var),str(type(var))))

def save_as_python( fd, var, varname, fname_extra=None ):
    if fname_extra is None:
        fname_extra = ''
    fname_prefix = varname + fname_extra
    buf = get_code_for_var( varname, fname_prefix, var)
    fd.write(buf)

if __name__ == '__main__':
    script_dir = os.path.abspath(os.path.split(__file__)[0])
    os.chdir(script_dir)
    test_repr()
########################################################
#    SAVE INFO (from save_sparse_weights.py)
########################################################

    receptor_dirs, tris = make_receptor_dirs(n_subdivides=1) # n_subidivisions
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


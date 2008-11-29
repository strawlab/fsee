# Copyright (C) 2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import os, math, warnings, sys, collections
import scipy.io

import numpy as np

import cgkit.cgtypes as cgtypes

import fsee
import fsee.Observer

def unique(A):
    """same as matlab's unique() function.

    not implemented for i

    A = B[j]
    """
    A = np.asarray(A)
    assert len(A.shape)==1
    B,i = np.unique1d( A,return_index=True )
    Blist = list(B)
    j = np.array([ Blist.index( Ai ) for Ai in A ])
    return B,i,j

def test_unique():
    A = [10,4,0,1,2,0,1,2,3,3,3]
    B_expected = [0,1,2,3,4,10]
    j_expected = [5,4,0,1,2,0,1,2,3,3,3]
    B,i,j = unique(A)
    A_new = B[j]
    assert np.allclose(B,B_expected)
    assert np.allclose(j,j_expected)
    assert np.allclose(A,A_new)

def mtrax_mat_to_big_arrays(data):
    """translation of code sent to me by alice"""
    if np.any(~np.isfinite(data['identity'])): # make sure no funny numbers
        raise ValueError('cannot handle non-finite data on identity')
    identity = np.array( data['identity'], dtype=int ) # cast to int
    assert np.allclose( identity, data['identity'] ) # double check
    idscurr,tmp,identity = unique(identity)

    if np.any(~np.isfinite(data['ntargets'])): # make sure no funny numbers
        raise ValueError('cannot handle non-finite data on ntargets')
    ntargets = np.asarray(data['ntargets'],dtype=int) # cast to int
    assert np.allclose( ntargets, data['ntargets'] )

    nframes = len(ntargets)
    nflies = max(identity)+1
    ## print 'nflies',nflies
    ## print 'nframes',nframes
    ## print 'identity.shape',identity.shape
    ## print 'identity[:5]',identity[:5]
    ## print 'identity[-5:]',identity[-5:]

    def nan(m,n):
        return np.nan * np.ones((m,n))
    X = nan(nflies,nframes)
    Y = nan(nflies,nframes)
    THETA = nan(nflies,nframes)
    startframe = nan(nflies,1)
    stopframe = nan(nflies,1)
    ## print 'X.shape',X.shape

    j=0
    for i in range(len(ntargets)):
        idx = j+np.arange(ntargets[i])
        id = identity[idx]
        ## print
        ## print 'i',i
        ## print 'idx',idx
        ## print 'id',id
        ## print 'X.shape',X.shape
        ## print 'X[id,i].shape',X[id,i].shape
        ## print "data['x_pos'][idx].shape",data['x_pos'][idx].shape
        X[id,i] = data['x_pos'][idx]
        Y[id,i] = data['y_pos'][idx]
        THETA[id,i] = data['angle'][idx]
        j += ntargets[i];

    X = X.T
    Y = Y.T
    THETA = THETA.T
    return X,Y,THETA, list(idscurr)

if __name__=='__main__':
    data = scipy.io.loadmat('movie20071009_154355.mat')
    #data = scipy.io.loadmat('newsave2.mat')
    print data.keys()

    X,Y,THETA,ids = mtrax_mat_to_big_arrays(data)

    if 0:
        import pylab
        pylab.figure()
        for j,id in enumerate(ids):
            pylab.plot(X[:,j],label='%d'%id)
        pylab.legend()
        pylab.xlabel('frame')
        pylab.ylabel('X')

        for j,id in enumerate(ids):
            pylab.figure()
            pylab.plot(X[:,j],Y[:,j])
            break

        pylab.show()
        #sys.exit(0)

    print '%d unique identities'%len(ids)

    xoffset = 500
    yoffset = 500

    radius_pix = 500

    # 10 inch = 25.4 cm = 254 mm = diameter = 2*radius
    pix_per_mm = 2.0*radius_pix/254.0

    mm_per_pixel = 1.0/pix_per_mm

    xgain = mm_per_pixel
    ygain = mm_per_pixel

    data_scale = 1
    xgain *= data_scale
    ygain *= data_scale

    X=(X-xoffset)*xgain
    Y=(Y-yoffset)*ygain

    if 0:
        import pylab
        pylab.plot(X[:,0],Y[:,0],'.')
        pylab.show()
        #sys.exit(0)

    if 0:
        pos_vec3,ori_quat = cgtypes.vec3(nums[0:3]),cgtypes.quat(nums[3:7])
        M = cgtypes.mat4().identity().translate(pos_vec3)
        M = M*ori_quat.toMat4()
        # grr, I hate cgtypes to numpy conversions!
        M = np.array((M[0,0],M[1,0],M[2,0],M[3,0],
                      M[0,1],M[1,1],M[2,1],M[3,1],
                      M[0,2],M[1,2],M[2,2],M[3,2],
                      M[0,3],M[1,3],M[2,3],M[3,3]))
        M.shape=(4,4)

    fly_model_node_filename = os.path.join(fsee.data_dir,'models/fly/body.osg')
    model_path = os.path.join(fsee.data_dir,"models/alice_cylinder/alice_cylinder.osg")
    #model_path = None
    z = 2 # 2 mm
    for j,id in enumerate(ids):
        print 'doing fly',id
        vision = fsee.Observer.Observer(model_path=model_path,
                                        scale=1000.0,
                                        hz=200.0,
                                        full_spectrum=True,
                                        optics='buchner71',
                                        do_luminance_adaptation=False,
                                        skybox_basename=None,
                                        )
        this_x = X[:,j]
        this_y = Y[:,j]
        this_theta = THETA[:,j]

        transformed_nodes_by_id = {}
        valid_rows = np.nonzero(~np.isnan(this_x))[0]
        #for i in range(valid_rows[0], valid_rows[-1]):
        #while 1:
        if 1:
            #for i in range(2000, 3000):
            #for i in range(2244, 2247):
            for i in range(2200, 2300):
                print 'frame',i
                # other flies
                all_existing_transforms = set(transformed_nodes_by_id.keys())
                this_frame_rendered = set()
                for other_id in ids:
                    if id==other_id:
                        continue # don't draw self
                    ## if other_id != 1:
                    ##     continue # skip for now
                    #print 'drawing',id
                    other_j = ids.index(other_id)
                    other_x = X[i,other_j]
                    other_y = Y[i,other_j]
                    other_theta = THETA[i,other_j]

                    if not np.isnan(other_x):
                        if other_id not in transformed_nodes_by_id:
                            transform = vision.sim.add_node_as_matrixtransform( fly_model_node_filename )
                            transformed_nodes_by_id[other_id] = transform
                        transform = transformed_nodes_by_id[other_id]

                        if 1:
                            pos_vec3 = cgtypes.vec3(( other_x, other_y, z ))
                            other_ori_quat = cgtypes.quat().fromAngleAxis(other_theta,(0,0,1))
                            if 1:
                                # pitch down
                                D2R = np.pi/180.0
                                other_ori_quat = other_ori_quat*cgtypes.quat().fromAngleAxis(80*D2R,(0,1,0))
                                #other_ori_quat = cgtypes.quat().fromAngleAxis(80*D2R,(1,0,0))*other_ori_quat
                                #other_ori_quat = other_ori_quat

                            M = cgtypes.mat4().identity().translate(pos_vec3)
                            M = M*other_ori_quat.toMat4()
                            # grr, I hate cgtypes to numpy conversions!
                            M = np.array((M[0,0],M[1,0],M[2,0],M[3,0],
                                          M[0,1],M[1,1],M[2,1],M[3,1],
                                          M[0,2],M[1,2],M[2,2],M[3,2],
                                          M[0,3],M[1,3],M[2,3],M[3,3]))
                            M.shape=(4,4)
                        transform.setMatrix(M)
                        this_frame_rendered.add( other_id )

                not_rendered = all_existing_transforms - this_frame_rendered
                if len(not_rendered):
                    print 'not rendered:',not_rendered
                    raise NotImplementedError('no implementation to deal with a disappearing fly')

                # now, do this fly
                ori_quat = cgtypes.quat().fromAngleAxis(this_theta[i],(0,0,1))
                posnow = cgtypes.vec3(( this_x[i], this_y[i], z))
                vision.step(posnow,ori_quat)
                if 1:
                    fname = 'fly%03d_frame%05d.png'%(id,i)
                    print 'saving', fname
                    vision.save_last_environment_map(fname)
                if 1:
                    fname = 'eye%03d_frame%05d.png'%(id,i)
                    R=vision.get_last_retinal_imageR()
                    G=vision.get_last_retinal_imageG()
                    B=vision.get_last_retinal_imageB()
                    #emds = vision.get_last_emd_outputs()
                    print 'saving', fname
                    fsee.plot_utils.plot_receptor_and_emd_fig(
                        R=R,G=G,B=B,#emds=emds,
                        save_fname=fname,
                        optics = vision.get_optics(),
                        proj='stere',
                        subplot_titles_enabled=False,
                        dpi=200)

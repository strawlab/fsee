# Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import math, sys
import numpy
import scipy
import scipy.sparse
import scipy.signal as signal
import ashelf

# NOTE: my coordinate system in phi is flipped from Lindemann's. In my
# system, phi increases to the left, but in Lindemann's, it increases
# to the right.


D2R = numpy.array(math.pi/180.0,dtype=numpy.float32)
R2D = numpy.array(180.0/math.pi,dtype=numpy.float32)

eye = 'left'

# coordinate system:
# azimuth: positive angles are left of center
# elevation: positive angles are above equator

# make receptor spatial positions for left eye
if eye == 'left':
    # left
    phi_deg = numpy.arange(120.0,-50.1,-2.0).astype(numpy.float32) # azimuth
else:
    # right
    phi_deg = numpy.arange(50.0,-120.1,-2.0).astype(numpy.float32) # azimuth
theta_deg = numpy.arange(-50.0,50.1,2.0).astype(numpy.float32) # elevation
phi = phi_deg*D2R
theta = theta_deg*D2R
n_receptors = len(phi_deg)*len(theta_deg)
print 'n_receptors',n_receptors

# transform image coordinates in pixels to phi and theta
imshape = 316,2048 # height x width in pixels
print 'imshape',imshape
imphi = -numpy.arange(imshape[1])/float(imshape[1])*2*math.pi # 360 degrees = 2 phi (rightward=negative)
imphi = imphi.astype(numpy.float32)
imphi_deg = imphi*R2D
impix_size = abs(imphi[1]-imphi[0]) # square pixels and cylindrical projection let us determine angular height
imtheta = numpy.arange(imshape[0])*impix_size
imtheta = imtheta - imtheta[ len(imtheta)//2 ] # center around equator
imtheta = imtheta.astype(numpy.float32)

# make receptor sensitivity matrix such that im*mat = receptors
imlen = imshape[0]*imshape[1] # m*n
im2receptors_shape = imlen, n_receptors

sigma=2.0*D2R
sigma2 = numpy.array(sigma**2,numpy.float32)
vert_gaussians = numpy.zeros( (len(theta),imshape[0]), dtype=numpy.float32 )
for i,theta_i in enumerate(theta):
    zetav = imtheta-theta_i
    zetav = numpy.mod(zetav+2*math.pi,2*math.pi)
    vert_gaussians[i,:] = numpy.exp(-zetav**2/sigma2)
horiz_gaussians = numpy.zeros( (len(phi),imshape[1]), dtype=numpy.float32 )
for j,phi_j in enumerate(phi):
    zetah = imphi-phi_j
    zetah = numpy.mod(zetah+2*math.pi,2*math.pi)
    horiz_gaussians[j,:] = numpy.exp(-zetah**2/sigma2)


SLOW_BUT_SAFE = 1
if SLOW_BUT_SAFE:
    im2receptors = scipy.sparse.dok_matrix((imlen,n_receptors), numpy.float32)
else:
    im2receptors = []
eps = 1e-4
print 'building matrix...'
nnz=0
for i,theta_i in enumerate(theta):
    for j,phi_j in enumerate(phi):
        
        R_idx = i*len(phi)+j # index into receptors
        
        vert = vert_gaussians[i,:]   # imshape[0]
        horiz = horiz_gaussians[j,:] # imshape[1]

        full_im = numpy.ravel(numpy.outer( vert, horiz ))
        sumim = numpy.sum(full_im)
        if sumim < eps:
            continue
        full_im = full_im / sumim # make sum to 1
        im_idxs = numpy.nonzero(full_im>eps)[0] # index into image
        if len(im_idxs):
            print 'R_idx %d significant entries (starts at %d) for receptor %d (of %d)'%(
                len(im_idxs),
                im_idxs[0],
                R_idx,
                n_receptors)
            if SLOW_BUT_SAFE:
                for im_idx in im_idxs:
                    im2receptors[int(im_idx),int(R_idx)] = full_im[im_idx]
                    if R_idx==(n_receptors-1):
                        print '-1,%d=%f'%(im_idx,full_im[im_idx])
            else:
                im2receptors.append( (R_idx,im_idxs,full_im[im_idxs]) )
                nnz+=len(im_idxs)
            
print 'converting to CSC...'
if SLOW_BUT_SAFE:
    im2receptors = im2receptors.tocsc()
else:
    print 'nnz',nnz
    data = numpy.zeros( (nnz,), dtype=numpy.float32 )
    inttype = numpy.uint32
    rowind = numpy.zeros( (nnz,), dtype=inttype )
    col_ptr = (nnz*numpy.ones( (im2receptors_shape[1]+1,))).astype( inttype )
    current_col = 0
    k = numpy.array(0,dtype=inttype)

    for R_idx,im_idxs,vals in im2receptors:
        ikey1 = R_idx
        while current_col <= ikey1:
            col_ptr[current_col]=int(k) # XXX TODO: int() is a scipy bug workaround
            current_col += 1
        for ikey0,val in zip(im_idxs,vals):
            data[k] = val
            rowind[k] = ikey0
            k += 1
            k = numpy.array(k,dtype=inttype) # XXX bugfix until scipy is fixed (promotes to Float64)
    im2receptors = scipy.sparse.csc_matrix((data, rowind, col_ptr),
                                           dims=im2receptors_shape,
                                           nzmax=nnz)
print 'done'

# transpose
im2receptors = im2receptors.transpose()

if 1:
    save_as_python = ashelf.save_as_python
    fd = open('cyl_proj.py','wb')
    fd.write( '# Automatically generated\n')
    fd.write( 'import numpy\n')
    fd.write( 'import scipy\n')
    fd.write( 'import scipy.sparse\n')
    fd.write( 'import os\n')
    fd.write( "__all__=['im2receptors','eye','phi','theta','imshape','imtheta','imphi']\n")
    fd.write( 'ashelf_datadir = os.path.split(__file__)[0]\n')
    save_as_python(fd, im2receptors, 'im2receptors')
    save_as_python(fd, eye, 'eye')
    save_as_python(fd, phi, 'phi')
    save_as_python(fd, theta, 'theta')
    save_as_python(fd, imshape, 'imshape')
    
    save_as_python(fd, imtheta, 'imtheta')
    save_as_python(fd, imphi, 'imphi')
    
    fd.write( '\n')
    fd.close()

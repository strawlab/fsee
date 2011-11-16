import pickle
import pylab
import scipy
import BugVision.ThesisStuff
import os

fd = open('all_results.pkl','rb')
all_results = pickle.load(fd)
fd.close()

pylab.figure()
ax = None

args_list = all_results.keys()
args_list.sort()

by_imfile = {}

for args in args_list:
    imfile, vel = args
    by_imfile.setdefault( imfile, {} )[vel] = all_results[args]

imfiles = by_imfile.keys()
imfiles.sort()

vels = by_imfile[ imfiles[0] ].keys()
vels.sort()

means_by_imfile = {}
for imfile in imfiles:
    means = [ scipy.average( by_imfile[imfile][vel] ) for vel in vels ]
    means_by_imfile[imfile] = means
    

im_codes = [ BugVision.ThesisStuff.get_image_code(os.path.split(imfile)[-1]) for imfile in imfiles ]
ims = zip(im_codes,imfiles)
ims.sort()

n_ax = len(ims)

ax1=pylab.subplot(n_ax,1,1)
for i in range(n_ax):
    im_code, imfile = ims[i]
    means = means_by_imfile[imfile]
    
    if i==0:
        ax = ax1
    else:
        ax = pylab.subplot(n_ax,1,1+i,sharex=ax1,sharey=ax1)

#    line, = ax.plot(vels,means)
    posvels = [-v for v in vels]
    line, = ax.semilogx(posvels,means,'o-')
    ax.legend([line],[im_code])
    pylab.setp( ax, 'ylim', [-1,11])
    pylab.setp( ax, 'yticks', [0,10])

    pylab.setp( ax, 'xticks', [10,100,1000])
        
    ax.set_ylabel( 'response\n(arbitrary)')
ax.set_xlabel('vel (deg/sec)')
pylab.show()

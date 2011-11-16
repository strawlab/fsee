import glob
import lindemann
import pickle
import scipy

SEPPO=0

if SEPPO:
    import seppo

if __name__=='__main__':

    imfiles = glob.glob('stim_images_fullsize/*')
    #imfiles = [ imfile.split[1] for imfile in imfiles ]
    #vels = -(10**scipy.linspace( scipy.log10(6.0), scipy.log10(2000.0),num=10))
    vels = -(10**scipy.linspace( scipy.log10(6.0), scipy.log10(2000.0),num=10))[1:-1]
    print 'vels',vels
    args = []
    for imfile in imfiles:
        for vel in vels:
            if 1:
                if not imfile.endswith('Glenelg.tif'):
                    continue
                if not -155<vel<-150:
                    continue
            args.append( (imfile, vel) )
    print 'args',args
    if SEPPO:
        Vms_list = seppo.map_parallel( lindemann.doit, args )
    else:
        Vms_list = map( lindemann.doit, args )

##    all_results = {}
##    for arg,Vms in zip(args,Vms_list):
##        all_results[arg] = Vms

##    fd = open('all_results.pkl','wb')
##    pickle.dump(all_results,fd)
##    fd.close()

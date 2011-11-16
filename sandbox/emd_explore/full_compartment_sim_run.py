import glob, os, sys
import scipy
import BugVision.ThesisStuff

if 1:
    imfiles = glob.glob('stim_images_fullsize/*')
    for imfile in imfiles:
        imcode = BugVision.ThesisStuff.get_image_code(os.path.split(imfile)[-1])
        print imcode,repr(imfile)
    imfiles = [ os.path.split(imfile)[1] for imfile in imfiles ]
    vels = -(10**scipy.linspace( scipy.log10(6.0), scipy.log10(2000.0),num=10))
    vels = list(vels)
    print repr(vels)
else:
    imfiles = [
        '080499-2048x316.bmp',
        'close_290102_2048x310.bmp',
        'gardens_20020508_2048x308.bmp',
        ]
if 1:
    vels = [ -151.26921505285739,
             
             -79.328764916291021,
             -288.44991406148154, 
             -41.601676461038082,
             -550.03493535021494,             
             -21.816796041072262,
             #-6.0,
             -11.441187711353814, 
             -1048.8421571906938,
             #-2000.0000000000002,
             ]

    n_cpus = 2

    queues = []
    for i in range(n_cpus):
        queues.append([])
        
    
    cnt = 0
    for vel_dps in vels:
        for imfile in imfiles:
            cmd = 'python full_compartment_sim.py %s %f'%(imfile,vel_dps)
            queues[cnt].append(cmd)
            cnt=(cnt+1)%n_cpus
            
    for i,q in enumerate(queues):
        print '#queue %d (of %d)'%(i+1,n_cpus)
        for cmd in q:
            print cmd

import glob, os, sys
import scipy
import BugVision.ThesisStuff
import tables

if 0:
    imfiles = glob.glob('stim_images_fullsize/*')
    for imfile in imfiles:
        imcode = BugVision.ThesisStuff.get_image_code(os.path.split(imfile)[-1])
        print imcode,repr(imfile)
    vels = -(10**scipy.linspace( scipy.log10(6.0), scipy.log10(2000.0),num=10))
    vels = list(vels)
    print repr(vels)
else:
    imfiles = [
        '080499-2048x316.bmp',
        'close_290102_2048x310.bmp',
        'gardens_20020508_2048x308.bmp',
        ]
    
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

    g_a = 20000.0

    all_sim_results = {}
    table_names = ['sim_results','sim_results_1_cpt']
    for table_name in table_names:
        print table_name
        rbi = {}
        for imfile in imfiles:
            imcode = BugVision.ThesisStuff.get_image_code(imfile)
            vs = []
            resps = []
            for vel_dps in vels:
                image_filename=imfile
                h5fname = 'sim_results_%s_%f.h5'%(image_filename,vel_dps)
                SAVEDIR = '/mnt/backup/andrew/lindemann2'
                #print 'opening',h5fname

                try:
                    h5file = tables.openFile(os.path.join(SAVEDIR,h5fname),mode='r')
                except IOError,err:
                    #print 'WARNING:',err
                    continue

                try:
                    sim_results = getattr(h5file.root,table_name)
                except tables.exceptions.NoSuchNodeError, err:
                    # haven't done this trial yet
                    #print 'WARNING:',err
                    continue

                all_Vs = sim_results.all_Vs
                times = sim_results.times
                g_a = sim_results.g_a[0]

                if table_name == 'sim_results':
                    emd_11= all_Vs[:,11]
                    neurite_0= all_Vs[:,24]
                neurite_last= all_Vs[:,-1]

                av = scipy.average( neurite_last )

                vs.append( vel_dps )
                resps.append( av )

                print imfile, 'vel_dps, av',vel_dps,av
            rbi[imcode] = (vs, resps)
        print table_name,id(rbi)
        all_sim_results[table_name] = rbi
        
    import pylab

    ax = None
    for i,table_name in enumerate(table_names):
        ax=pylab.subplot(len(table_names),1,i+1,sharex=ax,sharey=ax)
        rbi = all_sim_results[table_name]
        print table_name,id(rbi)
        if 1:
            lines = []
            titles = []
        imcodes = rbi.keys()
        imcodes.sort()
        fmts = dict(A='rx',
                    D='gx',
                    H='bx',)
        for imcode in imcodes:
            vs,resps = rbi[imcode]
            vs = -scipy.array(vs) # flip sign of velocities
            fmt = fmts[imcode]
            print vs, resps, fmt
            line,= ax.plot(vs,resps,fmt)
            if 1:
                lines.append(line)
                titles.append(imcode)
        if 1:
            pylab.legend(lines,titles)
        pylab.setp(ax,'ylim',[-1,5])
        ax.set_xscale('log')
        pylab.title(table_name)
    
    #pylab.savefig('cmpt_res.png')
    pylab.show()

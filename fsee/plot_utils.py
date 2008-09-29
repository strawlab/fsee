# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
try:
    from mpl_toolkits.basemap import Basemap # basemap > 0.9.9.1
except ImportError, err1:
    try:
        from matplotlib.toolkits.basemap import Basemap
    except ImportError, err2:
        raise err1

from matplotlib.collections import LineCollection
import pylab
import pickle
import math
import cgtypes # cgkit 1.x
import numpy

import fsee.eye_geometry.emd_util as emd_util
import fsee.eye_geometry.switcher
import drosophila_eye_map.util

# XXX I could refactor this to make basemap required only in a
# precomputation step.

#proj = 'moll'
#proj = 'cyl'
#proj = 'ortho'
proj = 'robin'

xyz2lonlat = drosophila_eye_map.util.xyz2lonlat # old name

class BasemapInstanceWrapper:
    def __init__(self,receptor_dirs=None,
                 edges=None,
                 hex_faces=None,
                 proj='ortho',
                 optics = None,
                 eye_name = None,
                 slicer = None,
                 edge_slicer=None,
                 **basemap_kws):

        if optics is not None:
            # automatically handle precomputed optics
            if (receptor_dirs is not None or
                edges is not None or
                hex_faces is not None):
                raise ValueError('if optics is specified, then '
                                 'receptor_dirs, edges, and hex_faces '
                                 'cannot be.')
            precomputed = fsee.eye_geometry.switcher.get_module_for_optics(optics=optics)

            self.rdirs = precomputed.receptor_dirs
            self.edges = precomputed.edges
            self.faces = precomputed.hex_faces
            self.slicer = precomputed.receptor_dir_slicer
            self.edge_slicer = precomputed.edge_slicer

        else:
            # manually specified optics
            self.rdirs = receptor_dirs
            self.edges = edges
            self.faces = hex_faces
            self.slicer = slicer
            self.edge_slicer = edge_slicer

        self.eye_name = eye_name


        supported_projs = ['moll','cyl','ortho','sinu',
                           'left_stere','right_stere',
                           'left_cyl','right_cyl',
                           ]
        if proj not in supported_projs:
            import warnings
            warnings.warn('projection %s not explicitly supported'%proj)

        # set defaults
        self.lon_scale = 1
        kws = dict(resolution=None)
        self.thresh_dist = 2e6
        self.magnitude2d_scale = 1.0 # rough normalization for 2D scaling

        # non-defaults
        if proj in ['moll','sinu','robin']:
            kws.update( dict(lon_0=0))
        elif proj  == 'ortho':
            kws.update( dict(lon_0=0.0,lat_0=0.0))
        elif proj == 'cyl':
            kws.update(dict(llcrnrlon=-180.,llcrnrlat=-90,
                        urcrnrlon=180.,urcrnrlat=90.))
            self.lon_scale=-1  # view from inside of eye
            self.thresh_dist=25
        elif proj == 'left_stere':
            proj = 'stere'
            kws.update(dict(lat_ts = 0.0,
                            lat_0 = 0,
                            lon_0 = 90,
                            llcrnrlon = -45,
                            urcrnrlon = -135,
                            llcrnrlat= -30,
                            urcrnrlat = 30,
                            ))
            self.magnitude2d_scale = 1e5 # rough normalization for 2D scaling
        elif proj == 'right_stere':
            proj = 'stere'
            kws.update(dict(lat_ts = 0.0,
                            lat_0 = 0,
                            lon_0 = -90,
                            llcrnrlon = 135,
                            urcrnrlon = 45,
                            llcrnrlat= -30,
                            urcrnrlat = 30,
                            ))
            self.magnitude2d_scale = 1e5 # rough normalization for 2D scaling
        elif proj == 'left_cyl':
            proj = 'cyl'
            kws.update(dict(llcrnrlon=0-40,llcrnrlat=-90,
                            urcrnrlon=180+10, urcrnrlat=90.))
            #self.lon_scale=-1  # view from inside of eye
            self.thresh_dist=25
        elif proj == 'right_cyl':
            proj = 'cyl'
            kws.update(dict(llcrnrlon=-180-10,llcrnrlat=-90,
                            urcrnrlon=0+40, urcrnrlat=90.))
            #self.lon_scale=-1  # view from inside of eye
            self.thresh_dist=25

        kws.update(basemap_kws)

        self.basemap_instance = Basemap(projection=proj, **kws)

        rdirs2 = [ self.xyz2lonlat( rdir.x, rdir.y, rdir.z ) for rdir in self.rdirs ]
        lons, lats = zip(*rdirs2)
        self.rdirs2_x, self.rdirs2_y = self.basemap_instance(lons, lats)

    def get_rdirs2xy(self,name=None):
        if name not in self.slicer:
            raise KeyError('no key %s in %s'%(name,str(self.slicer.keys())))
        return self.rdirs2_x[self.slicer[name]], self.rdirs2_y[self.slicer[name]]

    def xyz2lonlat(self,x,y,z):
        lon1,lat = xyz2lonlat(x,y,z)
        lon1 = lon1 * self.lon_scale
        return lon1,lat

    def get_visedges(self):
        return find_visible( self.edges, self.basemap_instance, self.rdirs2_x, self.rdirs2_y )

    def get_newfaces(self,close_polygons=False):
        vis_faces = get_vis_faces( self.faces, self )
        if close_polygons:
            for i,face in enumerate(vis_faces):
                if face is None:
                    continue
                xs, ys = list(face[0]), list(face[1])
                # make closed polygons
                xs.append(xs[0])
                ys.append(ys[0])
                vis_faces[i] = (xs,ys)
        return vis_faces


    def get_edges(self,name=None):
        return self.edges[self.edge_slicer[name]]

    def get_receptor_dirs(self,name=None):
        return self.rdirs[self.slicer[name]]

all_xdists = []

def find_visible( faces, m, rdirs2_x, rdirs2_y ):
    visfaces = []
    for face in faces:
        is_ok = True
        prev_projx = None
        for vi in face:
            projx = rdirs2_x[vi]
            projy = rdirs2_y[vi]

            if prev_projx is not None:
                xdist = abs(projx-prev_projx)
                all_xdists.append( xdist )
                if xdist > 250:
                    is_ok = False
                    break

            prev_projx = projx

            if projx < m.xmin or projx > m.xmax:
                is_ok = False
                break
            if projy < m.ymin or projy > m.ymax:
                is_ok = False
                break
        if is_ok:
            visfaces.append( face )
    return visfaces

def get_vis_faces( faces, wrapped_basemap_instance):
    newfaces = []
    for face in faces:
        newfacex = []
        newfacey = []
        is_ok = True
        for vert in face:
            #x,y,z = vert
            x,y,z = vert.x, vert.y, vert.z
            lon,lat = wrapped_basemap_instance.xyz2lonlat(x,y,z)

            projxs,projys = wrapped_basemap_instance.basemap_instance([lon],[lat])
            projx,projy = projxs[0],projys[0]

            newfacex.append( projx )
            newfacey.append( projy )

            if (projx < wrapped_basemap_instance.basemap_instance.xmin or
                projx > wrapped_basemap_instance.basemap_instance.xmax):
                is_ok = False
                break

            if (projy < wrapped_basemap_instance.basemap_instance.ymin or
                projy > wrapped_basemap_instance.basemap_instance.ymax):
                is_ok = False
                break

        if is_ok:
            newfaces.append( (newfacex, newfacey ) )
        else:
            newfaces.append( None )
    return newfaces

def plot_faces( wrapped_basemap_instance, ax, respsR, respsG, respsB ):
    newfaces = wrapped_basemap_instance.get_newfaces()
    assert len(respsR) == len(newfaces)
    assert len(respsG) == len(newfaces)
    assert len(respsB) == len(newfaces)

    slc = wrapped_basemap_instance.slicer[ wrapped_basemap_instance.eye_name ]
    for R,G,B,face in zip( respsR[slc],  respsG[slc], respsB[slc],
                           newfaces[slc] ):
        if face is not None:
            facex, facey = face
        else:
            continue

        xdist = max(facex) - min(facex)

        if 1:
            thresh = wrapped_basemap_instance.thresh_dist
            if xdist > thresh:
                #print 'skipping',facex
                continue

        r,g,b = R/255.0, G/255.0, B/255.0
        ax.fill( facex, facey, facecolor=(r,g,b), linewidth=0.01 )

def save_receptor_output_plot(new_filename,R,G,B,wrapped_basemap_instance=None,
                              optics = None, dpi=None, figsize=None):
    if wrapped_basemap_instance is not None:
        print 'WARNING: wrapped_basemap_instance argument no longer supported'
    plot_receptor_and_emd_fig(R=R,G=G,B=B,save_fname=new_filename,
                              optics=optics,dpi=dpi,figsize=figsize)

def save_receptor_output_plot_old(new_filename,R,G,B,wrapped_basemap_instance=None):
    if wrapped_basemap_instance is None:
        wrapped_basemap_instance = BasemapInstanceWrapper(proj='moll')

    import matplotlib
    matplotlib.use('Agg')
    import pylab

    fig = pylab.figure(figsize=(4,2))
    ax = fig.add_axes([0,0,1,1],frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plot_faces( wrapped_basemap_instance, ax, R, G, B )
    if 1:
        wrapped_basemap_instance.basemap_instance.drawmapboundary()
        # draw parallels and meridians.
        parallels = numpy.arange(-60.,90,30.)
        wrapped_basemap_instance.basemap_instance.drawparallels(parallels)#,labels=[1,0,0,0])
        meridians = numpy.arange(-360.,360.,30.)
        wrapped_basemap_instance.basemap_instance.drawmeridians(meridians)

    print 'saving',new_filename
    fig.savefig(new_filename)
    print 'done\n'
    pylab.close(fig)

def draw_arrow_xy_xy2(xy, xy2, edge_length=None, theta=None, fmt='k-'):
    lines = []
    x1,y1=xy
    x2,y2 = xy2
    lines.append( ((x1,y1), (x2,y2)) )

    if edge_length is None:
        edge_length = math.sqrt((y2-y1)**2 + (x2-x1)**2)
    if theta is None:
        theta = math.atan2( y2-y1, x2-x1 )

    arrow_angle = 30.0/math.pi*180
    arrow_scale = 0.3

    xd1 = arrow_scale*edge_length*math.cos(theta+arrow_angle)
    yd1 = arrow_scale*edge_length*math.sin(theta+arrow_angle)

    xa1 = x2+xd1
    ya1 = y2+yd1
    lines.append( ((x2,y2), (xa1,ya1)) )

    xd2 = arrow_scale*edge_length*math.cos(theta-arrow_angle)
    yd2 = arrow_scale*edge_length*math.sin(theta-arrow_angle)

    xa2 = x2+xd2
    ya2 = y2+yd2
    lines.append( ((x2,y2), (xa2,ya2)) )
    return lines

def plot_emd_outputs( wrapped_basemap_instance, ax, signal, scale=1.0,
                      threshold_magnitude=0.0, **kw):
    biw = wrapped_basemap_instance
    edges = biw.get_edges(name=biw.eye_name)
    emd_orig_dirs = emd_util.get_emd_center_directions(
        edges, biw.get_receptor_dirs() )
    mysignal = signal[biw.edge_slicer[biw.eye_name]]
    angular_vels = emd_util.project_emds_to_angular_velocity(
        mysignal,biw.get_receptor_dirs(),edges)
    plot_angular_vels( wrapped_basemap_instance, ax, emd_orig_dirs, angular_vels,
                       edges,
                       scale=scale, threshold_magnitude=threshold_magnitude, **kw)

def plot_angular_vels( wrapped_basemap_instance, ax,
                       emd_orig_dirs, angular_vels,
                       edges, scale=1.0, threshold_magnitude=0.0,
                       **kw):
    biw = wrapped_basemap_instance # shorthand

    method = '2Doffset' # 2D plotting of direction and magnitude from original direction
    #method = 'streamline' # velocity integration and backprojection onto sphere

    emd_orig_lons_lats = [biw.xyz2lonlat(eo.x,eo.y,eo.z) for eo in emd_orig_dirs]
    emd_orig_lons, emd_orig_lats = zip(*emd_orig_lons_lats)
    emd_orig_x, emd_orig_y = biw.basemap_instance(emd_orig_lons,emd_orig_lats)

    magnitudes = numpy.array([abs(v) for v in angular_vels])

    # find velocity (not angular velocity) at tangent point on sphere
    tangential_vels = [float(scale) * omega.cross(r) for omega,r in zip(angular_vels,emd_orig_dirs)]
    if method == '2Doffset':
        # XXX hack to minimize branch-cut failures
        tvs2 = []
        for tv in tangential_vels:
            if abs(tv) > 1e-10:
                tvs2.append( tv.normalize()*1e-10 )
            else:
                tvs2.append( tv )
        tangential_vels = tvs2

    # find position of particle traveling at that velocity for one time unit and project back onto sphere
    streamline_dirs = [v+r for v,r in zip(tangential_vels,emd_orig_dirs)]
    streamline_dirs = [d.normalize() for d in streamline_dirs]

    streamline_lons_lats = [biw.xyz2lonlat(sd.x,sd.y,sd.z) for sd in streamline_dirs]
    streamline_lons, streamline_lats = zip(*streamline_lons_lats)

    # XXX unfortunately, this step crosses branch cuts :(
    streamline_x, streamline_y = biw.basemap_instance(streamline_lons,streamline_lats)

    if method == 'streamline':
        emd_final_x, emd_final_y = streamline_x, streamline_y
    elif method == '2Doffset':
        streamline_delta_y = numpy.array(streamline_y)-numpy.array(emd_orig_y)
        streamline_delta_x = numpy.array(streamline_x)-numpy.array(emd_orig_x)
        direction_2D = numpy.arctan2(streamline_delta_y,streamline_delta_x)
        delta_x = scale*biw.magnitude2d_scale*magnitudes*numpy.cos(direction_2D)
        delta_y = scale*biw.magnitude2d_scale*magnitudes*numpy.sin(direction_2D)
        emd_final_x = emd_orig_x + delta_x
        emd_final_y = emd_orig_y + delta_y

    all_lines = []
    for i,edge in enumerate(edges):
        # assume emd_orig_dirs comes from same data
        assert len(emd_orig_dirs) == len( edges )
        if magnitudes[i]>threshold_magnitude:
            try:
                lines = draw_arrow_xy_xy2( (emd_orig_x[i],  emd_orig_y[i]),
                                           (emd_final_x[i], emd_final_y[i]),
                                           )
                all_lines.extend( lines )
            except OverflowError, err:
                raise
                #pass
    line_segments = LineCollection(all_lines,**kw)
    ax.add_collection(line_segments)
    #touchup_axes(biw,ax)

def touchup_axes(wrapped_basemap_instance,ax):
    biw = wrapped_basemap_instance # shorthand
    biw.basemap_instance.drawmapboundary(ax=ax)
    # draw parallels and meridians.
    parallels = numpy.arange(-60.,90,30.)
    biw.basemap_instance.drawparallels(parallels,ax=ax)#,labels=[1,0,0,0])
    meridians = numpy.arange(-360.,360.,30.)
    biw.basemap_instance.drawmeridians(meridians,ax=ax)

def plot_receptor_and_emd_fig( R=None,G=None,B=None,
                               emds=None, scale=1.0, title = None,
                               subplot_titles_enabled=True,
                               save_fname=None,
                               dpi=None, figsize=None,
                               overlay_receptor_circles=False,
                               emd_threshold_magnitude=0.0, # minimum magnitude to draw EMD arrow
                               emd_linewidth=0.3,
                               force_grid_lines = None,
                               optics = None,
                               proj = None):

    def do_overlay_receptor_circles(ax,biw,eye_name):
        rdirs2_x,rdirs2_y = biw.get_rdirs2xy(eye_name)
        ax.plot( rdirs2_x,rdirs2_y,'o',ms=0.6,
                 markeredgewidth=0.1,
                 #markerfacecolor='None',
                 markerfacecolor='white',
                 markeredgecolor='black')
    def draw_extra_lines(ax,biw):
        if draw_lines:  # grid
            basemap_lw = 0.1

            delat = 20.
            circles = numpy.arange(0.,90.,delat).tolist()+\
                      numpy.arange(-delat,-90,-delat).tolist()
            par = biw.basemap_instance.drawparallels(circles,ax=ax,
                                                     linewidth=basemap_lw)
            if 0:
                # for testing only:
                delon = 45.
                meridians = numpy.arange(0,180,delon)
            else:
                delon = 45.
                meridians = numpy.arange(-180,180,delon)

            mer = biw.basemap_instance.drawmeridians(meridians,ax=ax,
                                                     linewidth=basemap_lw)
    def get_new_xlim_ylim(zoom_factor,xlim,ylim):
        if 1:
            xmid = (xlim[0] + xlim[1])*0.5
            xdist = (xlim[1] - xlim[0])
            newxdist = xdist*zoom_factor
            xlim = xmid-newxdist*0.5,xmid+newxdist*0.5

            ymid = (ylim[0] + ylim[1])*0.5
            ydist = (ylim[1] - ylim[0])
            newydist = ydist*zoom_factor
            ylim = ymid-newydist*0.5,ymid+newydist*0.5
        return xlim,ylim
    def do_receptor_plot(ax,biw,eye_name,title):

        plot_faces( biw, ax, R,G,B)
        if title is not None:
            pylab.title(title,fontsize=10)
        ax.set_aspect('equal') # XXX newly added -- test
        ax.set_xticks([])
        ax.set_yticks([])
        draw_extra_lines(ax,biw)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        if flipX:
            xlim = (xlim[1],xlim[0])
        xlim,ylim = get_new_xlim_ylim(zoom_factor,xlim,ylim)
        if overlay_receptor_circles: do_overlay_receptor_circles(ax,biw,eye_name)
        ax.set_xlim( *xlim )
        ax.set_ylim( *ylim )
    def do_emd_plot(ax,biw,eye_name,title):
        plot_emd_outputs(biw, ax, emds, scale=scale,
                         threshold_magnitude=emd_threshold_magnitude,
                         linewidths=[emd_linewidth] )
        if title is not None:
            pylab.title(title,fontsize=10)
        draw_extra_lines(ax,biw)
        ax.set_aspect('equal') # XXX newly added -- test
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if flipX:
            xlim = xlim[1], xlim[0]
        xlim,ylim = get_new_xlim_ylim(zoom_factor,xlim,ylim)
        if overlay_receptor_circles: do_overlay_receptor_circles(ax,biw,eye_name)
        ax.set_xlim( *xlim )
        ax.set_ylim( *ylim )

    if optics not in ['synthetic','buchner71']:
        raise ValueError("optics argument must be 'synthetic' or 'buchner71'")

    plot_receptors = True
    if R is None and G is None and B is None:
        plot_receptors = False

    plot_emds = True
    if emds is None:
        plot_emds = False

    if figsize is None:
        if plot_emds and plot_receptors:
            figsize=(4,5)
        else:
            figsize=(4,2.5)

    if plot_emds and plot_receptors:
        fig = pylab.figure(figsize=figsize)
    else:
        fig = pylab.figure(figsize=figsize)

    if optics == 'synthetic':
        zoom_factor = 1.0
        if proj is None or proj == 'ortho':
            biw_front = BasemapInstanceWrapper(proj='ortho',lon_0=0.0,
                                               optics=optics)
            biw_back = BasemapInstanceWrapper(proj='ortho',lon_0=180.0,
                                              optics=optics)
        else:
            raise ValueError('for synthetic optics, only ortho projection known to work')

        biwA = biw_front
        biwB = biw_back
        if subplot_titles_enabled:
            titleA = 'front'
            titleB = 'back'
        else:
            titleA = titleB = None
        slicer_name_A = None
        slicer_name_B = None
        flipX = False
        draw_lines = False # grid

    elif optics == 'buchner71':
        zoom_factor = 0.85

        if proj == 'stere':
            biw_left  = BasemapInstanceWrapper(proj = 'left_stere',
                                               optics=optics,
                                               eye_name = 'left')
            biw_right = BasemapInstanceWrapper(proj = 'right_stere',
                                               optics=optics,
                                               eye_name = 'right')
        elif proj is None or proj == 'cyl':
            biw_left  = BasemapInstanceWrapper(proj = 'left_cyl',
                                               optics=optics,
                                               eye_name = 'left')
            biw_right = BasemapInstanceWrapper(proj = 'right_cyl',
                                               optics=optics,
                                               eye_name = 'right')

        #print 'biw_left.basemap_instance.ads_save_proj',biw_left.basemap_instance.ads_save_proj
        #print 'biw_right.basemap_instance.ads_save_proj',biw_right.basemap_instance.ads_save_proj

        biwA = biw_left
        biwB = biw_right
        if subplot_titles_enabled:
            titleA = 'left eye'
            titleB = 'right eye'
        else:
            titleA = titleB = None
        slicer_name_A = 'left'
        slicer_name_B = 'right'
        flipX = True
        draw_lines = True # grid (bug when True in basemap 0.9.2.dev-r2700)

    if force_grid_lines is not None:
        draw_lines = force_grid_lines

    xborder = 0.01
    left,mid,width = xborder, 0.5+xborder, 0.5-(2*xborder)

    if plot_receptors:
        if plot_emds:
            bottom, height = 0.5, 0.5
        else:
            bottom, height = 0.0,1.0

        ax = fig.add_axes([left,bottom,width,height],
                          frame_on=False)
        do_receptor_plot(ax,biwA,slicer_name_A,titleA)

        ax = fig.add_axes([mid,bottom,width,height],
                          frame_on=False)
        do_receptor_plot(ax,biwB,slicer_name_B,titleB)

    if plot_emds:
        if plot_receptors:
            bottom, height = 0.0, 0.5
        else:
            bottom, height = 0.0,1.0

        ax = fig.add_axes([left,bottom,width,height],
                          frame_on=False)
        do_emd_plot(ax,biwA,'left',titleA)

        ax = fig.add_axes([mid,bottom,width,height],
                          frame_on=False)
        do_emd_plot(ax,biwB,'right',titleB)

    if title is not None:
        pylab.figtext(0.5,0.95,title,
                      horizontalalignment='center',
                      verticalalignment='top',
                      fontsize=14)
    if save_fname is not None:
        if dpi is None:
            pylab.savefig(save_fname)
        else:
            pylab.savefig(save_fname,dpi=dpi)
    return fig


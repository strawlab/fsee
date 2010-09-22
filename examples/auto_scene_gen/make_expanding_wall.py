# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import fsee.scenegen.primlib as primlib
import math
import fsee.scenegen.osgwriter as osgwriter
import numpy
import numpy as np


D2R = np.pi/180.0
if 1:

    lambda_deg = 20.0
    wall_size = 160.0
    black_walls = []
    for i in range(wall_size//lambda_deg + 1):
        t0 = ((i-0.25)*lambda_deg-wall_size/2.0)*D2R
        t1 = ((i+0.25)*lambda_deg-wall_size/2.0)*D2R

        if abs(t0) <= lambda_deg*D2R*0.6:
            # this is where the post goes
            continue

        # on wall
        x = 1
        y0 = x*np.tan(t0)
        y1 = x*np.tan(t1)
        black_walls.append((y0,y1))

post_x = 0.5
post_y = 0
post_r = post_x*np.sin(lambda_deg*D2R/4.0)

if 1:
    import matplotlib.pyplot as plt
    for (y0,y1) in black_walls:
        plt.plot([x,x],[y0,y1],'k-')

        t0 = np.arctan2(y0,x)
        t1 = np.arctan2(y1,x)
        tt = np.linspace(t0,t1,10)
        plt.plot(np.cos(tt),np.sin(tt),'b-')

    plt.plot([0],[0],'rx')
    #plt.plot([0],[0],'ro')

    if 1:
        tt = np.linspace(0,2*np.pi, 30 )
        plt.plot( post_x + post_r*np.cos(tt), post_y+post_r*np.sin(tt), 'k-' )

    plt.gca().set_aspect('equal')
    plt.gca().set_xlim([-1,2])
    plt.show()

# units in m
height = 20.0

z0 = 0
z1 = height

if 1:
    # make walls
    geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])

    bwalls2 = []
    for bw in black_walls:
        bwalls2.extend( bw )

    x = 1.0

    color = 'white'
    for i0 in range(len(bwalls2)-1):
        y0 = bwalls2[i0]
        y1 = bwalls2[i0+1]
        if color=='white':
            color = 'black'
        elif color=='black':
            color = 'white'

        wall = primlib.XZRect()
        wall.texture_fname = '%s.png'%color
        wall.mag_filter = "NEAREST"
        z0 = 0
        z1 = height
        wall.verts = [[ x, y0, z0],
                      [ x, y0, z1],
                      [ x, y1, z1],
                      [ x, y1, z0]]
        geode.append(wall.get_as_osg_geometry())

m = osgwriter.MatrixTransform(numpy.eye(4))
m.append(geode)

post = primlib.ZCyl()
post.verts += np.array([0,0,1000]) # goes from 0 to 2000
post.verts /= np.array([1,1,2000]) # goes from 0 to 1
post.verts *= np.array([1,1,height]) # goes from 0 to height

post.verts *= np.array([post_r,post_r,1])
post.verts += np.array([post_x,post_y,0])

post.texture_fname = 'black.png'
post.mag_filter = "NEAREST"

pm = osgwriter.MatrixTransform(numpy.eye(4))
geode2 = osgwriter.Geode(states=['GL_LIGHTING OFF'])
geode2.append(post.get_as_osg_geometry())
pm.append( geode2 )
m.append(pm)

g = osgwriter.Group()
g.append(m)

fd = open('expanding_wall.osg','wb')
g.save(fd)
fd.close()


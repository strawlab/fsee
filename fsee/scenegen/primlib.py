# Copyright (C) 2005-2007 California Institute of Technology,
# All rights reserved
# Author: Andrew D. Straw
import osgwriter
import math
import numpy

class Prim:
    def __init__(self):
        self.texture_fname = "texture.jpg"
        self.mag_filter = "LINEAR"
        self.verts = []
        self.normals = []
        self.tex_coords = []
        self.prim_sets = []

    def get_as_osg_geometry(self):
        ss = osgwriter.StateSet(self.texture_fname)
        ss.mag_filter = self.mag_filter
        verts = osgwriter.Array(name="VertexArray",
                                points = self.verts)
        normals = osgwriter.Array(name="NormalArray",
                                points = self.normals)
        tex_coords = osgwriter.Array(name="TexCoordArray 0 Vec2Array",
                                     points = self.tex_coords)
        prim_sets = osgwriter.PrimitiveSets()
        for prim_set in self.prim_sets:
            prim_sets.append( prim_set.get_as_osg_prim_set() )

        geom = osgwriter.Geometry(ss,
                                  verts,
                                  normals,
                                  prim_sets,
                                  tex_coords)
        return geom

class Quads:
    def __init__(self,faces):
        self.faces = faces
    def get_as_osg_prim_set(self):
        return osgwriter.Quads(self.faces)

class XYRect(Prim):
    def __init__(self):
        Prim.__init__(self)
        self.verts = [[ 0, 0, 0],
                      [ 0, 1, 0],
                      [ 1, 1, 0],
                      [ 1, 0, 0]]
        self.normals = [[0,0,1],
                        [0,0,1],
                        [0,0,1],
                        [0,0,1]]
        self.prim_sets = [Quads( [[0,1,2,3]] )]
        self.tex_coords = [[0,0],
                           [0,1],
                           [1,1],
                           [1,0]]

class XZRect(Prim):
    def __init__(self):
        Prim.__init__(self)
        self.verts = [[ 0, 0, 0],
                      [ 0, 0, 1],
                      [ 1, 0, 1],
                      [ 1, 0, 0]]
        self.normals = [[0,1,0],
                        [0,1,0],
                        [0,1,0],
                        [0,1,0]]
        self.prim_sets = [Quads( [[0,1,2,3]] )]
        self.tex_coords = [[0,0],
                           [0,1],
                           [1,1],
                           [1,0]]

class YZRect(Prim):
    def __init__(self):
        Prim.__init__(self)
        self.verts = [[ 0, 0, 0],
                      [ 0, 0, 1],
                      [ 0, 1, 1],
                      [ 0, 1, 0]]
        self.normals = [[1,0,0],
                        [1,0,0],
                        [1,0,0],
                        [1,0,0]]
        self.prim_sets = [Quads( [[0,1,2,3]] )]
        self.tex_coords = [[0,0],
                           [0,1],
                           [1,1],
                           [1,0]]

class ZCyl(Prim):
    def __init__(self,res=32):
        Prim.__init__(self)

        fracs = numpy.linspace( 0.0, 1.0, res+1 )
        start_frac = fracs[:-1]
        stop_frac = fracs[1:]

        angles = fracs*360.0
        starts = angles[:-1]
        stops = angles[1:]

        D2R = math.pi/180.0
        start_x = numpy.cos(starts*D2R)
        start_y = numpy.sin(starts*D2R)
        stop_x = numpy.cos(stops*D2R)
        stop_y = numpy.sin(stops*D2R)

        self.verts = []
        self.normals = []
        quads = []
        count = 0
        self.tex_coords = []

        z0=-1000
        z1=1000

        for x0,y0,x1,y1,f0,f1 in zip(start_x,start_y,
                                     stop_x,stop_y,
                                     start_frac,stop_frac):
            self.verts.extend( [ [x0,y0,z0],
                                 [x0,y0,z1],
                                 [x1,y1,z1],
                                 [x1,y1,z0],
                                 ])
            self.normals.extend( [[x0,y0,0],
                                  [x0,y0,0],
                                  [x1,y1,0],
                                  [x1,y1,0],
                                  ])
            self.tex_coords.extend( [ [f0,0],
                                      [f0,1],
                                      [f1,1],
                                      [f1,0],
                                      ])

            quads.append( [count,count+1,count+2,count+3] )
            count+=4

        self.prim_sets = [Quads(quads)]

def test():
    import sys
    import scipy

    geode = osgwriter.Geode()

    geode.append(XYRect().get_as_osg_geometry())
    geode.append(XZRect().get_as_osg_geometry())

    m = osgwriter.MatrixTransform(scipy.eye(4))
    m.append(geode)

    g = osgwriter.Group()
    g.append(m)
    g.save(sys.stdout)

if __name__ == '__main__':
    test()

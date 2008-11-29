# Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
class Saves:
    def __init__(self):
        self.spi = 2
        
class HasChildren(Saves):
    def __init__(self):
        Saves.__init__(self)
        self.children=[]
        self.children_name = 'children'
    def append(self,val):
        self.children.append(val)
    def write_children(self,fd,indent):
        istr = ' '*indent
        fd.write( '%snum_%s %d\n'%(istr,self.children_name,len(self.children),))
        for child in self.children:
            child.save(fd,indent=indent)

class Array(Saves):
    def __init__(self,name='Array',points=[]):
        Saves.__init__(self)
        self.name = name
        self.points = points
    def save(self,fd,indent=0):
        istr = ' '*indent
        fd.write( '%s%s %d {\n'%(istr,self.name,len(self.points)))
        indent += self.spi
        istr = ' '*indent

        for i in range(len(self.points)):
            fd.write( '%s%s\n'%(istr,' '.join(map(repr,self.points[i]))))

        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))
    
class Geode(HasChildren):
    def __init__(self,states=None):
        HasChildren.__init__(self)
        self.myname='Geode'
        self.children_name = 'drawables'
        self.states = states
        
    def save(self,fd,indent=0):
        istr = ' '*indent
        fd.write( '%s%s {\n'%(istr,self.myname))
        indent += self.spi
        istr = ' '*indent
        
        fd.write( '%sDataVariance DYNAMIC\n'%(istr,))
        fd.write( '%scullingActive TRUE\n'%(istr,))
        if self.states is not None:
            fd.write( '%sStateSet {\n'%(istr,))
            for state in self.states:
                fd.write( '%s  %s\n'%(istr,state,))
            fd.write( '%s}\n'%(istr,))

        self.write_children(fd,indent)

        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))

class BillboardChild(Saves):
    def __init__(self,positions=None):
        self.positions = positions
        Saves.__init__(self)
    def save(self,fd,indent=0):
        istr = ' '*indent
        if 1:
            fd.write( '%sMode AXIAL_ROT\n'%(istr,))
            fd.write( '%sAxis 0 0 1\n'%(istr,))
            fd.write( '%sNormal 0 -1 0\n'%(istr,))
        elif 0:
            fd.write( '%sMode POINT_ROT_WORLD\n'%(istr,))
        else:
            fd.write( '%sMode POINT_ROT_EYE\n'%(istr,))

        if self.positions is not None:
            fd.write( '%sPositions {\n'%(istr,))
            
            indent += self.spi
            istr2 = ' '*indent
            for p in self.positions:
                fd.write( '%s%f %f %f\n'%(istr2,p[0],p[1],p[2]))
            fd.write( '%s}\n'%(istr,))
            
        
        #indent += self.spi
        #istr = ' '*indent
        
class Billboard(Geode):
        
    def __init__(self,*args,**kws):
        if 'positions' in kws:
            positions = kws['positions']
            del kws['positions']
        else:
            positions = None
        Geode.__init__(self,*args,**kws)
        self.myname='Billboard'
        self.children.append(BillboardChild(positions=positions))

class StateSet(Saves):
    def __init__(self,texturename):
        Saves.__init__(self)
        assert isinstance(texturename,str)
        self.texturename = texturename
        self.mag_filter = "LINEAR"
        
    def save(self,fd,indent=0):
        istr = ' '*indent
        fd.write( '%sStateSet {\n'%(istr,))
        
        indent += self.spi
        istr = ' '*indent

        fd.write( '%sDataVariance STATIC\n'%(istr,))
        fd.write( '%srendering_hint DEFAULT_BIN\n'%(istr,))
        fd.write( '%srenderBinMode INHERIT\n'%(istr,))
        
        fd.write( '%stextureUnit 0 {\n'%(istr,))
        indent += self.spi
        istr = ' '*indent
        fd.write( '%sGL_TEXTURE_2D ON\n'%(istr,))

        fd.write( '%sTexture2D {\n'%(istr,))
        indent += self.spi
        istr = ' '*indent
        fd.write( '%sDataVariance STATIC\n'%(istr,))
        fd.write( '%sfile "%s"\n'%(istr,self.texturename))
        fd.write( '%swrap_s REPEAT\n'%(istr,))
        fd.write( '%swrap_t REPEAT\n'%(istr,))
        fd.write( '%swrap_r CLAMP\n'%(istr,))
        fd.write( '%smin_filter LINEAR_MIPMAP_LINEAR\n'%(istr,))
        fd.write( '%smag_filter %s\n'%(istr,self.mag_filter))
        fd.write( '%smaxAnisotropy 1\n'%(istr,))
        fd.write( '%sinternalFormatMode USE_IMAGE_DATA_FORMAT\n'%(istr,))
        
        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))

        
        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))
        


        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))

class Geometry(Saves):
    def __init__(self,
                 state_set,
                 vertex_array,
                 normal_array,
                 primitive_sets,
                 tex_coord_array):
        Saves.__init__(self)
        assert isinstance(state_set,StateSet)
        self.state_set = state_set
        self.vertex_array = vertex_array
        self.normal_array = normal_array
        self.primitive_sets = primitive_sets
        self.tex_coord_array = tex_coord_array
        
    def save(self,fd,indent=0):
        istr = ' '*indent
        fd.write( '%sGeometry {\n'%(istr,))
        
        indent += self.spi
        istr = ' '*indent

        self.state_set.save(fd,indent=indent)
        self.vertex_array.save(fd,indent=indent)
        fd.write( '%sNormalBinding PER_VERTEX\n'%(istr,))
        self.normal_array.save(fd,indent=indent)
        self.primitive_sets.save(fd,indent=indent)
        self.tex_coord_array.save(fd,indent=indent)
        
        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))
    
class Group(HasChildren):
    def save(self,fd,indent=0):
        istr = ' '*indent
        fd.write( '%sGroup {\n'%(istr,))
        
        indent += self.spi
        istr = ' '*indent
        fd.write( '%sUniqueID osgwriterGroup_%X\n'%(istr,abs(id(self))))
        fd.write( '%sDataVariance DYNAMIC\n'%(istr,))
        fd.write( '%sname "osgwriterFile"\n'%(istr,))
        fd.write( '%scullingActive TRUE\n'%(istr,))

        self.write_children(fd,indent)
        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))
        
class MatrixTransform(HasChildren):
    def __init__(self,M):
        HasChildren.__init__(self)
        assert len(M)==4
        for i in range(4):
            assert len(M[i]) == 4
        self.M = M
        
    def save(self,fd,indent=4):
        istr = ' '*indent
        fd.write( '%sMatrixTransform {\n'%(istr,))
        
        indent += self.spi
        istr = ' '*indent
        
        fd.write( '%sDataVariance DYNAMIC\n'%(istr,))
        fd.write( '%scullingActive TRUE\n'%(istr,))
        fd.write( '%sreferenceFrame RELATIVE_TO_PARENTS\n'%(istr,))
        fd.write( '%sname "osgwriterMatrixTransform"\n'%(istr,))

        fd.write( '%sMatrix {\n'%(istr,))
        indent += self.spi
        istr = ' '*indent
        fd.write( '%sDataVariance DYNAMIC\n'%(istr,))
        for i in range(4):
            fd.write( '%s%s\n'%(istr,' '.join(map(repr,self.M[i]))))
        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))

        self.write_children(fd,indent)

        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))

class Faces(Saves):
    def __init__(self,faces,typename):
        Saves.__init__(self)
        self.faces = faces
        self.typename = typename
    def save(self,fd,indent=4):
        istr = ' '*indent
        fd.write( '%sDrawElementsUInt %s %d {\n'%(istr,self.typename,len(self.faces)))
        
        indent += self.spi
        istr = ' '*indent

        for face in self.faces:
            fd.write( '%s%s\n'%(istr,' '.join(map(repr,face))))

        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))
        
class Quads(Faces):
    def __init__(self,quads):
        Faces.__init__(self,quads,'QUADS')
class Tris(Faces):
    def __init__(self,tris):
        Faces.__init__(self,tris,'TRIANGLES')
        
class PrimitiveSets(Saves):
    def __init__(self):
        Saves.__init__(self)
        self.prim_sets = []
    def append(self,prims):
        self.prim_sets.append(prims)
    def save(self,fd,indent=4):
        istr = ' '*indent
        fd.write( '%sPrimitiveSets %d {\n'%(istr,len(self.prim_sets)))
        
        indent += self.spi
        istr = ' '*indent

        for child in self.prim_sets:
            child.save(fd,indent)

        indent -= self.spi
        istr = ' '*indent
        fd.write( '%s}\n'%(istr,))
        
    
def test():
    import sys
    import scipy

    ss = StateSet("random.png")
    verts = Array(name="VertexArray",
                  points = scipy.zeros((8,3)))
    normals = Array(name="NormalArray",
                    points = scipy.zeros((8,3)))
    tex_coords = Array(name="TexCoordArray 0 Vec2Array",
                       points = scipy.zeros((8,2)))
    prim_sets = PrimitiveSets()
    quads = Quads([[0,1,2,3],[5,6,7,4]])
    prim_sets.append(quads)
    
    geom = Geometry(ss,
                    verts,
                    normals,
                    prim_sets,
                    tex_coords)
    #geom.append()
    
    geode = Geode()
    geode.append(geom)
    
    m = MatrixTransform(scipy.eye(4))
    m.append(geode)
    
    g = Group()
    g.append(m)
    g.save(sys.stdout)

if __name__ == '__main__':
    test()

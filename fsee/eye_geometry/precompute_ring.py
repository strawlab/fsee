
from precompute_generic import precompute_generic
import cgtypes
from math import cos, sin, pi
import numpy

def deg2rad(x):
    return x * pi / 180
    
def precompute_ring_configuration(conf_name, fov_degrees, num_receptors):
    fov = deg2rad(fov_degrees)
    thetas = numpy.linspace( -fov/2, +fov/2, num_receptors)
    
    receptor_dirs = [ cgtypes.vec3(cos(theta), 0.01*sin(100*theta), sin(theta) ) 
        for theta in thetas]

    tris = [(n,n+1,n+2) for n in range(0, num_receptors-2)]
    # BUG: otherwise pseudo_voronoi does not terminate (?)
    tris.reverse()
    precompute_generic(conf_name, receptor_dirs, tris)
    
if __name__ == "__main__":
    
    fov_degrees = [350, 180]
    num_receptors = [60, 90, 180]
     
    for fov in fov_degrees:
        for num in num_receptors:
            conf_name = 'ring_fov%d_num%d' % (fov, num)
            precompute_ring_configuration(conf_name, fov, num)
    

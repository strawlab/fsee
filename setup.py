# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved

import os, glob, sys, time, shutil
from setuptools import setup, find_packages
from setuptools.dist import Distribution

fsee_package_data = [
    'data/models/alice_cylinder/*',
    'data/models/mamarama_checkerboard/*',
    ]

if 1:
    fsee_package_data.extend([
    'data/models/fly/wing.osg',
    'data/models/fly/wingim.jpg',

    'data/models/fly/body.osg',
    'data/models/fly/brownredwhite.png',

    'data/models/floor_only/floor_only.osg',
    'data/models/floor_only/rocks.jpg',

    'data/models/WT1/WT1.osg',
    'data/models/WT1/post_black.png',
    'data/models/WT1/tunnel_floor_texture.png',
    'data/models/WT1/tunnel_wall_inside_texture.png',
    'data/models/WT1/vent.png',

    'data/models/tunnel_one_wall/tunnel.osg',
    'data/models/tunnel_one_wall/random.png',
    'data/models/tunnel_one_wall/rocks.jpg',

    'data/models/tunnel_leftturn/tunnel.osg',
    'data/models/tunnel_leftturn/random.png',
    'data/models/tunnel_leftturn/rocks.jpg',

    'data/models/tunnel_rightturn/tunnel.osg',
    'data/models/tunnel_rightturn/random.png',
    'data/models/tunnel_rightturn/rocks.jpg',

    'data/models/tunnel_straight/random.png',
    'data/models/tunnel_straight/random2.png',
    'data/models/tunnel_straight/rocks.jpg',
    'data/models/tunnel_straight/tunnel_straight_large_texture.osg',
    'data/models/tunnel_straight/tunnel_straight_small_texture.osg',

    'data/Images/brightday1_cubemap/negx.png',
    'data/Images/brightday1_cubemap/negy.png',
    'data/Images/brightday1_cubemap/negz.png',
    'data/Images/brightday1_cubemap/posx.png',
    'data/Images/brightday1_cubemap/posy.png',
    'data/Images/brightday1_cubemap/posz.png',

    'data/models/grating_tunnel/WT_grating_0.1000floormpc_0.1000wallmpc.osg',
    'data/models/grating_tunnel/grating_16_cycles.png',

    ])

if sys.platform == 'win32':
    prefix = 'fsoi_ng'
    extension = '.dll'
elif sys.platform.startswith('linux'):
    prefix = 'libfsoi_ng'
    extension = '.so'
elif sys.platform.startswith('darwin'):
    prefix = 'libfsoi_ng'
    extension = '.dylib'

libfile = os.path.join('src','fsoi',prefix+extension)
if not os.path.exists(libfile):
    raise RuntimeError('libfsoi_ng (or fsoi_ng.dll) does not exist -- aborting -- run scons in src/fsoi')

# copy .so or .dll from build directory into target directory
shutil.copy2(libfile,os.path.join('fsee',prefix+extension))

fsee_package_data.append( prefix+extension)

package_data = {'fsee':fsee_package_data}

package_data['fsee.eye_geometry']=['*.mat','*.m']

eager_files = [ 'fsee/'+f for f in fsee_package_data ]

if 1:
    kwargs = dict(
        zip_safe=False,
        eager_resources = eager_files,
        entry_points = {'console_scripts': [
        'fsee_monitor = fsee.fsee_monitor:main',
        'fsee_envmap2bugeye = fsee.envmap2bugeye:main',
        'fsee_bench = fsee.fsee_bench.main',
        'fsee_fsoi_runner = fsee.FlySimWrapFSOI:real_runner',
        ]
                        },
        )
else:
    kwargs = {}

class PlatformDependentDistribution(Distribution):
    # Force platform-dependant build.
    def has_ext_modules(self):
        return True

setup(name="fsee",
      description='model of fly visual system',
      author="Andrew Straw",
      author_email="strawman@astraw.com",
      version='0.2', # keep in sync with fsee/__init__.py
      packages = find_packages(),
      package_data = package_data,
      distclass = PlatformDependentDistribution,
      **kwargs)

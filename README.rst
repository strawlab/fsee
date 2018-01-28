fsee - simulation of a fly eye view
-----------------------------------

This software is part of GUF, the Grand Unified Fly. For an overview,
see https://strawlab.org/2011/03/23/grand-unified-fly/

Installation
------------

To install, one must run the non-standard step of building fsoi with scons before running the standard python setup routine::

    cd src/fsoi
    scons
    cd ../../fsee/eye_geometry
    make
    cd ../..
    python setup.py install

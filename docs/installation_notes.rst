Installing on OS X
==================

(AC) I can confirm that now fsee works on OS X (Leopard). However, it took several steps of which I lost track. But: do try, it is indeed possible.



Installing on Ubuntu Lucid
========================

Install dependencies: ::

	$ sudo apt-get install libopenscenegraph-dev

Also, get cgkit from Andrew's PPA https://launchpad.net/~astraw/+archive/ppa

Compile ``fsoi``: ::

	$ cd src/fsoi
	$ scons

Notes about ``basemap``
----------------------

The ``basemap`` library is needed only by the realtime plotter;
after recent changes, you can use fsee without it.

It is a pain to install ``basemap`` on Mac/Lucid because of recursive dependencies.
I suggest you don't even try.

If you do, note tha ``basemap`` requires a specific version ``geoslib`` 

Cannot import is

Install basemap: install geoslib inside the basemap distribution (2.2.3).  Install geos: compilation fails because it misses <string.h>



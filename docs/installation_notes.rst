Installing on OS X
==================

Leopard (10.5)
--------------

(AC) I can confirm that now fsee works on OS X (Leopard). However, it
took several steps of which I lost track. But: do try, it is indeed
possible.


Snow Leopard (10.6)
-------------------

You can install some required libraries using Macports:

	$ sudo port install OpenSceneGraph scons

Installing on Ubuntu Lucid
==========================

Install dependencies::

	$ sudo apt-get install libopenscenegraph-dev

Also, get cgkit from Andrew's PPA https://launchpad.net/~astraw/+archive/ppa

Compile ``fsoi``::

	$ cd src/fsoi
	$ scons

Notes about ``basemap``
-----------------------

The ``basemap`` library is needed only by the realtime plotter;
after recent changes, you can use fsee without it.

It is a pain to install ``basemap`` on Mac because of recursive
dependencies.  I suggest you don't even try. It's easier on
Lucid. Install ``libgeos-dev``.

If you do, note that ``basemap`` requires a specific version ``geoslib``

Cannot import is

Install basemap: install geoslib inside the basemap distribution
(2.2.3).  Install geos: compilation fails because it misses <string.h>



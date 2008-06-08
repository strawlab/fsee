# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved

import os

__all__ = ['monitor_file']
from pkg_resources import Requirement, resource_filename # from setuptools

data_dir = resource_filename(__name__,"data") # trigger extraction
default_skybox = os.path.join(data_dir,'Images/brightday1_cubemap/')

__version__ = 0.2 # keep in sync with setup.py

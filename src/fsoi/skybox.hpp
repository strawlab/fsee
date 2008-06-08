// Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
// Author: Andrew D. Straw
#ifndef _SKYBOX_HPP
#define _SKYBOX_HPP

#ifdef _MSC_VER // this is MS compiler specific, but how else to test for Windows?
#include <windows.h>
#endif

#include <osg/Node>
#include <osg/ClearNode>

//void add_skybox_to_node(osg::ClearNode&);

void add_skybox_to_node(osg::ref_ptr<osg::ClearNode> mynode,
			std::string posx_fname,
			std::string posy_fname,
			std::string posz_fname,
			std::string negx_fname,
			std::string negy_fname,
			std::string negz_fname
			);
#endif

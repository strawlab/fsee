// Copyright (C) 2005-2007 California Institute of Technology, All rights reserved
// Author: Andrew D. Straw
#include "skybox.hpp"

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Quat>
#include <osg/Matrix>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/Transform>
#include <osg/Material>
#include <osg/NodeCallback>
#include <osg/Depth>
#include <osg/CullFace>
#include <osg/TexMat>
#include <osg/TexGen>
#include <osg/TexEnvCombine>
#include <osg/TextureCubeMap>
#include <osg/VertexProgram>

#include <osg/TexEnv>

#include <osgDB/Registry>
#include <osgDB/ReadFile>

#include <osgUtil/SmoothingVisitor>
#include <osgUtil/Optimizer>
#include <osgUtil/CullVisitor>

osg::TextureCubeMap* readCubeMap(std::string posx_fname,
				 std::string negx_fname,
				 std::string posy_fname,
				 std::string negy_fname,
				 std::string posz_fname,
				 std::string negz_fname)
{
  osg::TextureCubeMap* cubemap = new osg::TextureCubeMap;

    osg::Image* imagePosX = osgDB::readImageFile(posx_fname);
    osg::Image* imageNegX = osgDB::readImageFile(negx_fname);
    osg::Image* imagePosY = osgDB::readImageFile(posy_fname);
    osg::Image* imageNegY = osgDB::readImageFile(negy_fname);
    osg::Image* imagePosZ = osgDB::readImageFile(posz_fname);
    osg::Image* imageNegZ = osgDB::readImageFile(negz_fname);

    if (imagePosX && imageNegX && imagePosY && imageNegY && imagePosZ && imageNegZ)
    {
        cubemap->setImage(osg::TextureCubeMap::POSITIVE_X, imagePosX);
        cubemap->setImage(osg::TextureCubeMap::NEGATIVE_X, imageNegX);
        cubemap->setImage(osg::TextureCubeMap::POSITIVE_Y, imagePosY);
        cubemap->setImage(osg::TextureCubeMap::NEGATIVE_Y, imageNegY);
        cubemap->setImage(osg::TextureCubeMap::POSITIVE_Z, imagePosZ);
        cubemap->setImage(osg::TextureCubeMap::NEGATIVE_Z, imageNegZ);

        cubemap->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
        cubemap->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
        cubemap->setWrap(osg::Texture::WRAP_R, osg::Texture::CLAMP_TO_EDGE);

        cubemap->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
        cubemap->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    }

  return cubemap;
}

// Update texture matrix for cubemaps
struct TexMatCallback : public osg::NodeCallback
{
public:

    TexMatCallback(osg::TexMat& tm) :
        _texMat(tm)
    {
    }

    virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
    {
        osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>(nv);
        if (cv)
        {
	    const osg::Matrix& MV = *(cv->getModelViewMatrix());
            const osg::Matrix R = osg::Matrix::rotate( osg::DegreesToRadians(112.0f), 0.0f,0.0f,1.0f)*
                                  osg::Matrix::rotate( osg::DegreesToRadians(90.0f), 1.0f,0.0f,0.0f);

            osg::Quat q;
            MV.get(q);
            const osg::Matrix C = osg::Matrix::rotate( q.inverse() );

            _texMat.setMatrix( C*R );
        }

        traverse(node,nv);
    }

    osg::TexMat& _texMat;
};

class MoveEarthySkyWithEyePointTransform : public osg::Transform
{
public:
    /** Get the transformation matrix which moves from local coords to world coords.*/
    virtual bool computeLocalToWorldMatrix(osg::Matrix& matrix,osg::NodeVisitor* nv) const 
    {
        osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>(nv);
        if (cv)
        {
            osg::Vec3 eyePointLocal = cv->getEyeLocal();
            matrix.preMult(osg::Matrix::translate(eyePointLocal));
        }
        return true;
    }

    /** Get the transformation matrix which moves from world coords to local coords.*/
    virtual bool computeWorldToLocalMatrix(osg::Matrix& matrix,osg::NodeVisitor* nv) const
    {
        osgUtil::CullVisitor* cv = dynamic_cast<osgUtil::CullVisitor*>(nv);
        if (cv)
        {
            osg::Vec3 eyePointLocal = cv->getEyeLocal();
            matrix.postMult(osg::Matrix::translate(-eyePointLocal));
        }
        return true;
    }
};

void add_skybox_to_node(osg::ref_ptr<osg::ClearNode> mynode,
			std::string posx_fname,
			std::string negx_fname,
			std::string posy_fname,
			std::string negy_fname,
			std::string posz_fname,
			std::string negz_fname
			)
{

    osg::StateSet* stateset = new osg::StateSet();

    osg::TexEnv* te = new osg::TexEnv;
    te->setMode(osg::TexEnv::REPLACE);
    stateset->setTextureAttributeAndModes(0, te, osg::StateAttribute::ON);

    osg::TexGen *tg = new osg::TexGen;
    tg->setMode(osg::TexGen::NORMAL_MAP);
    stateset->setTextureAttributeAndModes(0, tg, osg::StateAttribute::ON);

    osg::TexMat *tm = new osg::TexMat;
    stateset->setTextureAttribute(0, tm);

    osg::TextureCubeMap* skymap = readCubeMap(posx_fname,
					      negx_fname,
					      posy_fname,
					      negy_fname,
					      posz_fname,
					      negz_fname);

    stateset->setTextureAttributeAndModes(0, skymap, osg::StateAttribute::ON);

    stateset->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
    stateset->setMode( GL_CULL_FACE, osg::StateAttribute::OFF );

    // clear the depth to the far plane.
    osg::Depth* depth = new osg::Depth;
    depth->setFunction(osg::Depth::ALWAYS);
    depth->setRange(1.0,1.0);   
    stateset->setAttributeAndModes(depth, osg::StateAttribute::ON );

    stateset->setRenderBinDetails(-1,"RenderBin");

    osg::ref_ptr<osg::Drawable> drawable = new osg::ShapeDrawable( new osg::Sphere(osg::Vec3(0.0f,0.0f,0.0f),500));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->setCullingActive(false);
    geode->setStateSet( stateset );
    geode->addDrawable(drawable.get());

    osg::ref_ptr<osg::Transform> transform = new MoveEarthySkyWithEyePointTransform();
    transform->setCullingActive(false);
    transform->addChild(geode.get());

//  mynode.setRequiresClear(false);
    mynode->setCullCallback(new TexMatCallback(*tm));
    mynode->addChild(transform.get());
}

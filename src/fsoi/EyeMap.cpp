#include "EyeMap"
#include <osg/Geometry>
#include <assert.h>

/** constructor */
EyeMap::EyeMap( ) {
  // Vertices
  _verts = new osg::Vec3Array;

  for (int i=0; i<2; i++) {
    osg::Vec3 base_pos = osg::Vec3(i*30, 0.0f, 0.0f);

    _verts->push_back( base_pos );                                      // center
    _verts->push_back( base_pos + osg::Vec3( -10.0f, -10.0f, 0.0f ) );  // edge v0
    _verts->push_back( base_pos + osg::Vec3( -10.0f,  10.0f, 0.0f ) );  // edge v1
    _verts->push_back( base_pos + osg::Vec3(  10.0f,  10.0f, 0.0f ) );  // edge v2
    _verts->push_back( base_pos + osg::Vec3(  10.0f, -10.0f, 0.0f ) );  // edge v3
    _verts->push_back( base_pos + osg::Vec3( -10.0f, -10.0f, 0.0f ) );  // edge v0
  }


  // Colours
  _colors = new osg::Vec4Array;
  _colors->push_back( osg::Vec4( 1.0f, 0.0f, 0.0f, 1.0f ));
  _colors->push_back( osg::Vec4( 0.0f, 1.0f, 0.0f, 1.0f ));


  _fan_lengths = new osg::ByteArray;
  _fan_lengths->push_back( 6 );
  _fan_lengths->push_back( 6 );

  createDrawables();
}

void EyeMap::createDrawables() {
  // Remove any existing Drawables
  _drawables.erase(_drawables.begin(), _drawables.end());

  // 1. First the faces
  faces = new osg::Geometry();
  faces->setSupportsDisplayList(false);

  faces->setVertexArray(_verts.get());
  faces->setColorArray(_colors.get());
  faces->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);

  // Normal
  osg::ref_ptr<osg::Vec3Array> ns(new osg::Vec3Array);
  ns->push_back(osg::Vec3(0.0f,0.0f,1.0f));
  faces->setNormalArray(ns.get());
  faces->setNormalBinding(osg::Geometry::BIND_OVERALL);

  int cur_size = 0;
  GLbyte this_len = 0;
  for (int i=0; i<_fan_lengths->getNumElements(); i++) {
    this_len = _fan_lengths->index(i);
    // The triangle fan that represents the face
    faces->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN,cur_size,this_len));
    cur_size += this_len;
  }
  
  addDrawable(faces.get());
}

void EyeMap::setGeometry( osg::Vec3Array* verts, osg::ByteArray* fan_lengths ) {
  _verts = verts;
  _fan_lengths = fan_lengths;


  // Colours
  _colors = new osg::Vec4Array;
  for (int i=0; i<fan_lengths->getNumElements(); i++ ) {
    _colors->push_back( osg::Vec4( 1.0f, 1.0f, 1.0f, 1.0f ));
  }

  createDrawables();
}

void EyeMap::setColors(osg::Vec4Array* cs) {
  assert(cs->getNumElements() == _colors->getNumElements());

  _colors = cs;
  // XXX dirty the drawables rather than this?
  faces->setColorArray(_colors.get());
}

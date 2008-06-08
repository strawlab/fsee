extern "C" {
#include "fsoi_ng.h"
}

#ifdef USE_SKYBOX
#include "skybox.hpp"
#endif

#if 1
#define DPRINTF(...)
#else
#define DPRINTF(...) printf(__VA_ARGS__)
#endif

#include <osg/GLExtensions>
#include <osg/Node>
#include <osg/Geometry>
#include <osg/Notify>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/Stencil>
#include <osg/ColorMask>
#include <osg/Depth>
#include <osg/Billboard>
#include <osg/Material>
#include <osg/AnimationPath>

#include <osgGA/TrackballManipulator>

#include <osgUtil/SmoothingVisitor>

#include <osgDB/Registry>
#include <osgDB/ReadFile>

#include <osgViewer/Viewer>

#include <iostream>

#include "EyeMap"

// XXX on ATI linux drivers, this callback is required for RTT to work.
class MyGeometryCallback :
    public osg::Drawable::UpdateCallback,
    public osg::Drawable::AttributeFunctor
{
    public:

        MyGeometryCallback(const osg::Vec3& o,
                           const osg::Vec3& x,const osg::Vec3& y,const osg::Vec3& z,
                           double period,double xphase,double amplitude) {}


        virtual void update(osg::NodeVisitor* nv,osg::Drawable* drawable)
        {
        }

        virtual void apply(osg::Drawable::AttributeType type,unsigned int count,osg::Vec3* begin)
        {
        }

};

struct MyCameraPostDrawCallback : public osg::Camera::DrawCallback
{
    MyCameraPostDrawCallback(osg::Image* image):
        _image(image)
    {
    }

    virtual void operator () (const osg::Camera& /*camera*/) const
    {
      /*        if (_image && _image->getPixelFormat()==GL_RGBA && _image->getDataType()==GL_UNSIGNED_BYTE)
        {
            // we'll pick out the center 1/2 of the whole image,
            int column_start = _image->s()/4;
            int column_end = 3*column_start;

            int row_start = _image->t()/4;
            int row_end = 3*row_start;


            // and then invert these pixels
            for(int r=row_start; r<row_end; ++r)
            {
                unsigned char* data = _image->data(column_start, r);
                for(int c=column_start; c<column_end; ++c)
                {
                    (*data) = 255-(*data); ++data;
                    (*data) = 255-(*data); ++data;
                    (*data) = 255-(*data); ++data;
                    (*data) = 255; ++data;
                }
            }


            // dirty the image (increments the modified count) so that any textures
            // using the image can be informed that they need to update.
            _image->dirty();
        }
        else if (_image && _image->getPixelFormat()==GL_RGBA && _image->getDataType()==GL_FLOAT)
        {
            // we'll pick out the center 1/2 of the whole image,
            int column_start = _image->s()/4;
            int column_end = 3*column_start;

            int row_start = _image->t()/4;
            int row_end = 3*row_start;

            // and then invert these pixels
            for(int r=row_start; r<row_end; ++r)
            {
                float* data = (float*)_image->data(column_start, r);
                for(int c=column_start; c<column_end; ++c)
                {
                    (*data) = 1.0f-(*data); ++data;
                    (*data) = 1.0f-(*data); ++data;
                    (*data) = 1.0f-(*data); ++data;
                    (*data) = 1.0f; ++data;
                }
            }

            // dirty the image (increments the modified count) so that any textures
            // using the image can be informed that they need to update.
            _image->dirty();
        }
      */
    }

    osg::Image* _image;
};


osg::Node* createPreRenderSubGraph(osg::Node* subgraph, unsigned tex_width, unsigned tex_height, osg::Camera::RenderTargetImplementation renderImplementation, bool useImage, bool useTextureRectangle, bool useHDR, osg::Image* image, osg::Camera* camera)
{
    if (!subgraph) return 0;

    // create a group to contain the flag and the pre rendering camera.
    osg::Group* parent = new osg::Group;

    // texture to render to and to use for rendering of flag.
    osg::Texture* texture = 0;
    if (useTextureRectangle)
    {
        osg::TextureRectangle* textureRect = new osg::TextureRectangle;
        textureRect->setTextureSize(tex_width, tex_height);
        textureRect->setInternalFormat(GL_RGBA);
        textureRect->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR);
        textureRect->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::LINEAR);

        texture = textureRect;
    }
    else
    {
        osg::Texture2D* texture2D = new osg::Texture2D;
        texture2D->setTextureSize(tex_width, tex_height);
        texture2D->setInternalFormat(GL_RGBA);
        texture2D->setFilter(osg::Texture2D::MIN_FILTER,osg::Texture2D::LINEAR);
        texture2D->setFilter(osg::Texture2D::MAG_FILTER,osg::Texture2D::LINEAR);

        texture = texture2D;
    }

    if (useHDR)
    {
        texture->setInternalFormat(GL_RGBA16F_ARB);
        texture->setSourceFormat(GL_RGBA);
        texture->setSourceType(GL_FLOAT);
    }

    // first create the geometry of the flag of which to view.
    {
      // XXX ATI linux driver (fglrx) seems buggy in that RTT doesn't work unless this actually happens.
      // XXX nvidia linux driver (nvidia 100.14.19 on GeForce 6600/PCI/SSE2 amd64) seems buggy in that RTT is screwed up unless this happens.

        // create the to visualize.
        osg::Geometry* polyGeom = new osg::Geometry();

        polyGeom->setSupportsDisplayList(false);

        osg::Vec3 origin(0.0f,0.0f,0.0f);
        osg::Vec3 xAxis(1.0f,0.0f,0.0f);
        osg::Vec3 yAxis(0.0f,0.0f,1.0f);
        osg::Vec3 zAxis(0.0f,-1.0f,0.0f);
        float height = 100.0f;
        float width = 200.0f;
        int noSteps = 20;

        osg::Vec3Array* vertices = new osg::Vec3Array;
        osg::Vec3 bottom = origin;
        osg::Vec3 top = origin; top.z()+= height;
        osg::Vec3 dv = xAxis*(width/((float)(noSteps-1)));

        osg::Vec2Array* texcoords = new osg::Vec2Array;

        // note, when we use TextureRectangle we have to scale the tex coords up to compensate.
        osg::Vec2 bottom_texcoord(0.0f,0.0f);
        osg::Vec2 top_texcoord(0.0f, useTextureRectangle ? tex_height : 1.0f);
        osg::Vec2 dv_texcoord((useTextureRectangle ? tex_width : 1.0f)/(float)(noSteps-1),0.0f);

        for(int i=0;i<noSteps;++i)
        {
            vertices->push_back(top);
            vertices->push_back(bottom);
            top+=dv;
            bottom+=dv;

            texcoords->push_back(top_texcoord);
            texcoords->push_back(bottom_texcoord);
            top_texcoord+=dv_texcoord;
            bottom_texcoord+=dv_texcoord;
        }


        // pass the created vertex array to the points geometry object.
        polyGeom->setVertexArray(vertices);

        polyGeom->setTexCoordArray(0,texcoords);


        osg::Vec4Array* colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(1.0f,1.0f,1.0f,1.0f));
        polyGeom->setColorArray(colors);
        polyGeom->setColorBinding(osg::Geometry::BIND_OVERALL);

        polyGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUAD_STRIP,0,vertices->size()));

        // new we need to add the texture to the Drawable, we do so by creating a
        // StateSet to contain the Texture StateAttribute.
        osg::StateSet* stateset = new osg::StateSet;

        stateset->setTextureAttributeAndModes(0, texture,osg::StateAttribute::ON);

        polyGeom->setStateSet(stateset);

	polyGeom->setUpdateCallback(new MyGeometryCallback(origin,xAxis,yAxis,zAxis,1.0,1.0/width,0.2f));

        osg::Geode* geode = new osg::Geode();
        geode->addDrawable(polyGeom);

        parent->addChild(geode);
    }


    // then create the camera node to do the render to texture
    {
        // set up the background color and clear mask.
        camera->setClearColor(osg::Vec4(0.1f,0.1f,0.3f,1.0f));
        camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        const osg::BoundingSphere& bs = subgraph->getBound();
        if (!bs.valid())
        {
            return subgraph;
        }

        float znear = 1.0f*bs.radius();
        float zfar  = 3.0f*bs.radius();

        // 2:1 aspect ratio as per flag geomtry below.
        float proj_top   = 0.25f*znear;
        float proj_right = 0.5f*znear;

        znear *= 0.9f;
        zfar *= 1.1f;

        // set up projection.
        camera->setProjectionMatrixAsFrustum(-proj_right,proj_right,-proj_top,proj_top,znear,zfar);

        // set view
        camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
        camera->setViewMatrixAsLookAt(bs.center()-osg::Vec3(0.0f,2.0f,0.0f)*bs.radius(),bs.center(),osg::Vec3(0.0f,0.0f,1.0f));

        // set viewport
	camera->setViewport(0,0,tex_width,tex_height);

        // set the camera to render before the main camera.
        camera->setRenderOrder(osg::Camera::PRE_RENDER);

        // tell the camera to use OpenGL frame buffer object where supported.
        camera->setRenderTargetImplementation(renderImplementation);


        if (useImage)
        {
	  //            osg::Image* image = new osg::Image;
            //image->allocateImage(tex_width, tex_height, 1, GL_RGBA, GL_UNSIGNED_BYTE);
	  //image->allocateImage(tex_width, tex_height, 1, GL_RGBA, GL_FLOAT);

            // attach the image so its copied on each frame.
            camera->attach(osg::Camera::COLOR_BUFFER, image);

	    /*
            camera->setPostDrawCallback(new MyCameraPostDrawCallback(image));

            // Rather than attach the texture directly to illustrate the texture's ability to
            // detect an image update and to subload the image onto the texture.  You needn't
            // do this when using an Image for copying to, as a seperate camera->attach(..)
            // would suffice as well, but we'll do it the long way round here just for demonstation
            // purposes (long way round meaning we'll need to copy image to main memory, then
            // copy it back to the graphics card to the texture in one frame).
            // The long way round allows us to mannually modify the copied image via the callback
            // and then let this modified image by reloaded back.
            texture->setImage(0, image);
	    */
        }
        else
        {
            // attach the texture and use it as the color buffer.
            camera->attach(osg::Camera::COLOR_BUFFER, texture);
        }


        // add subgraph to render
        camera->addChild(subgraph);

        parent->addChild(camera);

    }

    return parent;
}


class MyFsoiObj {
public:
  // constructor
  MyFsoiObj(const char* filename,
	    double scale,
	    const char *skybox_basename,
	    double im_xang,
	    double im_yang,
	    double near,
	    double far,
	    unsigned tex_width=64,
	    unsigned tex_height=64,
	    //osg::Camera::RenderTargetImplementation renderImplementation = osg::Camera::FRAME_BUFFER_OBJECT,
	    osg::Camera::RenderTargetImplementation renderImplementation = osg::Camera::FRAME_BUFFER,
	    bool useTextureRectangle = false,
	    bool useHDR = false
	    ) {
    DPRINTF("constructor called\n");

    std::vector<std::string> filenames = std::vector<std::string>();
    filenames.push_back( std::string(filename) );

    bool useImage = true;

    osg::Node* loadedModel = osgDB::readNodeFiles(filenames);
    if (!loadedModel)
    {
      DPRINTF("Aborting: could not load files\n");
      exit(2);
    }
    osg::MatrixTransform* loadedModelTransform = new osg::MatrixTransform;
    loadedModelTransform->addChild(loadedModel);

#define SEIZE_CAMERA
#ifdef SEIZE_CAMERA
    DPRINTF("setting scale to %f.\n",scale);
    loadedModelTransform->setMatrix( osg::Matrix::scale(scale,scale,scale) );
#else
    // spin the model.
    osg::NodeCallback* nc = new osg::AnimationPathCallback(loadedModelTransform->getBound().center(),osg::Vec3(0.0f,0.0f,1.0f),osg::inDegrees(45.0f));
    loadedModelTransform->setUpdateCallback(nc);
#endif

    prerender_image = new osg::Image;
    DPRINTF("allocating image...\n");
    prerender_image->allocateImage(tex_width, tex_height, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    //prerender_image->allocateImage(tex_width, tex_height, 1, GL_RGB, GL_UNSIGNED_BYTE);
    DPRINTF("allocated (%dx%d) OK.\n",tex_width,tex_height);
    //image->allocateImage(tex_width, tex_height, 1, GL_RGBA, GL_FLOAT);

    prerender_camera = new osg::Camera;


  osg::Group* model_and_sky = new osg::Group();
#ifdef USE_SKYBOX
  if (skybox_basename) {
    osg::ref_ptr<osg::ClearNode> skybox_node = new osg::ClearNode();
    std::string basedir = std::string(skybox_basename);
    // weird step to prevent strlen symbol lookup problem on Ubuntu Feisty
    std::string px = std::string(basedir+"posx.png");
    std::string nx = std::string(basedir+"negx.png");
    std::string ny = std::string(basedir+"negy.png");
    std::string py = std::string(basedir+"posy.png");
    std::string pz = std::string(basedir+"posz.png");
    std::string nz = std::string(basedir+"negz.png");
    add_skybox_to_node( skybox_node, px, nx, ny,py, pz,nz);
    model_and_sky->addChild( skybox_node.get() );
  }
#else
  if (skybox_basename) {
    fprintf(stderr,"error: requested skybox, but support not compiled in %s:%d\n",__FILE__,__LINE__);
  }
#endif
  model_and_sky->addChild( loadedModelTransform );

    rootNode = new osg::Group();
    rootNode->addChild(createPreRenderSubGraph(model_and_sky,tex_width,tex_height, renderImplementation, useImage, useTextureRectangle, useHDR, prerender_image.get(), prerender_camera.get()));

#ifdef SEIZE_CAMERA
    // set camera properties (re-setting what was done in createPreRenderSubGraph)
    prerender_camera->setProjectionMatrixAsPerspective( im_yang, im_xang/im_yang, near, far);
#endif

    // add model to the viewer.
    viewer.setSceneData( rootNode.get() );

#if 1
    // render to window (not fullscreen)

    osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
    traits->x = 0;
    traits->y = 0;
    traits->width = 640;
    traits->height = 480;
    traits->red = 8;
    traits->green = 8;
    traits->blue = 8;
    traits->alpha = 8;
    traits->depth = 32;
    traits->windowDecoration = true;
    traits->doubleBuffer = true;
    //traits->doubleBuffer = false;

    osg::ref_ptr<osg::GraphicsContext> gc = osg::GraphicsContext::createGraphicsContext(traits.get());

    osg::ref_ptr<osg::Camera> camera = viewer.getCamera();

    camera->setGraphicsContext(gc.get());
    camera->setViewport(new osg::Viewport(0,0, traits->width, traits->height));
    GLenum buffer = traits->doubleBuffer ? GL_BACK : GL_FRONT;
    camera->setDrawBuffer(buffer);
    camera->setReadBuffer(buffer);

    // set background color
    camera->setClearColor(osg::Vec4(0.0f,0.0f,0.0f,1.0f));
    camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#endif

    viewer.setKeyEventSetsDone(0); // disable keypress from quitting
    viewer.setQuitEventSetsDone(false); // disable quit event from quitting (XXX seems broken as of 2007-06-27)

    if (!viewer.getCameraManipulator() && viewer.getCamera()->getAllowEventFocus())
    {
      viewer.setCameraManipulator(new osgGA::TrackballManipulator());
      //        viewer.setCameraManipulator(new osgGA::MatrixManipulator());
    }

    if (!viewer.isRealized()) {
      viewer.realize();
    }

    addEyeMap(0);
    addEyeMap(1);

    DPRINTF("constructor finished\n");
  }

  // destructor
  ~MyFsoiObj() {
    DPRINTF("destructor called\n");
  }

  void setEyeMapGeometry(int num, osg::Vec3Array* verts, osg::ByteArray* fan_lengths ) {
    eye_map_geode[num]->setGeometry( verts, fan_lengths );
  }

  void addEyeMap(int num) {
    // this part inspired by osgscalarbar.cpp createScalarBar_HUD() function
    eye_map_geode[num] = new EyeMap();

    osg::StateSet * stateset = eye_map_geode[num]->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    stateset->setMode(GL_DEPTH_TEST,osg::StateAttribute::OFF);
    stateset->setRenderBinDetails(11, "RenderBin");

    osg::MatrixTransform * modelview = new osg::MatrixTransform;

    modelview->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    modelview->addChild(eye_map_geode[num].get());

    _eyemap_projection[num] = new osg::Projection;
    _eyemap_projection[num]->setMatrix(osg::Matrix::ortho2D(-40,40,-30,30)); // x1 x2 y1 y2
    _eyemap_projection[num]->addChild(modelview);
    rootNode->addChild(_eyemap_projection[num].get());
  }

  void setEyeMapProjection(int num,float x1,float x2,float y1,float y2) {
    _eyemap_projection[num]->setMatrix( osg::Matrix::ortho2D(x1,x2,y1,y2));
  }

  void run() {
    // the mainloop
    while (!viewer.done())
    {
        viewer.frame();
    }
  }

  void set_eyemap_face_colors( int num, osg::Vec4Array* cs ) {
    eye_map_geode[num]->setColors(cs);
  }

  void render_frame(int* OK, unsigned char** image_data, int* s, int* t, int* r,
		    GLenum* pixelFormat, GLenum* type,
		    unsigned int* packing) {
    // the mainloop
    if (viewer.done()) {
      *OK = 0;
      return;
    }

    viewer.frame();
    *image_data = prerender_image->data();
    *s = prerender_image->s();
    *t = prerender_image->t();
    *r = prerender_image->r();
    *pixelFormat = prerender_image->getPixelFormat();
    *type =  prerender_image->getDataType();
    *packing = prerender_image->getPacking();

    *OK = 1;
  }

  void set_pos_ori(double x,double y,double z,
		   double qw,double qx,double qy,double qz) {
    osg::Vec3d position = osg::Vec3d( x, y, z );
    osg::Quat rotation = osg::Quat( qx, qy, qz, qw );

    osg::Matrix m;
    m.preMult(osg::Matrixd::rotate(rotation.inverse()));
    m.preMult(osg::Matrixd::translate(-position));
    prerender_camera->setViewMatrix(m);
  }

  inline void get_width_height(int* width,int* height) {
    *width = prerender_image->s();
    *height = prerender_image->t();
  }

private:
  // construct the viewer.
  osgViewer::Viewer viewer;
  osg::ref_ptr<osg::Group> rootNode;
  osg::ref_ptr<osg::Image> prerender_image;
  osg::ref_ptr<osg::Camera> prerender_camera;
  osg::ref_ptr<EyeMap> eye_map_geode[2];
  osg::ref_ptr<osg::Projection> _eyemap_projection[2];
};

extern "C" {

FsoiErr fsoi_ng_init() {
  DPRINTF("initializing\n");
  return FsoiNoErr;
}

FsoiErr fsoi_ng_new(FsoiObj** theobjptr, const char* filename, double scale,
		    const char* skybox_basename,
		    double im_xang, double im_yang, double near, double far,
		    int im_width, int im_height, const char* render_implementation) {
  DPRINTF("new called (scale %f, near %f, far %f, render %s)\n",scale, near, far, render_implementation);

  osg::Camera::RenderTargetImplementation renderImplementation;

  if (!strcmp(render_implementation,"fbo")) {
    renderImplementation= osg::Camera::FRAME_BUFFER_OBJECT;
  } else if (!strcmp(render_implementation,"pbuffer")) {
    renderImplementation= osg::Camera::PIXEL_BUFFER;
  } else if (!strcmp(render_implementation,"pbuffer-rtt")) {
    renderImplementation= osg::Camera::PIXEL_BUFFER_RTT;
  } else if (!strcmp(render_implementation,"fb")) {
    renderImplementation= osg::Camera::FRAME_BUFFER;
  } else if (!strcmp(render_implementation,"window")) {
    renderImplementation= osg::Camera::SEPERATE_WINDOW;
  } else {
    fprintf(stderr,"unknown render target: %s\n",render_implementation);
    return FsoiUnknownRenderImplementeation;
  }

  *theobjptr = new FsoiObj;

  FsoiObj* theobj = (*theobjptr);
  MyFsoiObj* the_cpp_obj = new MyFsoiObj(filename,scale,
					 skybox_basename,
					 im_xang,
					 im_yang,
					 near,
					 far,
					 im_width,
					 im_height);
  theobj->the_cpp_obj = (void*)(the_cpp_obj);
  DPRINTF("new done\n");

  return FsoiNoErr;
}

FsoiErr fsoi_ng_delete(FsoiObj* theobj) {
  DPRINTF("delete called\n");
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;
  delete mfo;
  delete theobj;

  return FsoiNoErr;
}

FsoiErr fsoi_ng_run(FsoiObj* theobj) {
  DPRINTF("running\n");
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;
  mfo->run();
  return FsoiNoErr;
}

FsoiErr fsoi_ng_render_frame(FsoiObj* theobj,
			     unsigned char** image_data_ptr,
			     int* width,
			     int* height,
			     int* num_bytes_per_pixel) {
  DPRINTF("rendering frame\n");
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;
  int OK;

  int r;
  unsigned int packing;
  GLenum pixelFormat, type;

  mfo->render_frame(&OK,image_data_ptr,width,height,&r,&pixelFormat,&type,&packing);
  if (!OK) {
    return FsoiRenderFrameError;
  }

  if (*image_data_ptr && pixelFormat==GL_RGBA && type==GL_UNSIGNED_BYTE && r==1) {
    *num_bytes_per_pixel=4;
    DPRINTF("RGBA width %d, height %d\n",*width,*height);
    return FsoiNoErr;
  }
  else if (*image_data_ptr && pixelFormat==GL_RGB && type==GL_UNSIGNED_BYTE && r==1) {
    *num_bytes_per_pixel=3;
    DPRINTF("RGB width %d, height %d\n",*width,*height);
    return FsoiNoErr;
  }

  return FsoiUnsupportedData;
}

FsoiErr fsoi_ng_render_frame_copy(FsoiObj* theobj,
				  unsigned char* image_data,
				  int width,
				  int height,
				  int num_bytes_per_pixel) {
  unsigned char* orig_data_ptr;
  FsoiErr e;
  int s,t,src_bpp;

  e = fsoi_ng_render_frame(theobj, &orig_data_ptr, &s, &t, &src_bpp);
  if (e != FsoiNoErr) {
    return e;
  }

  if (!(s==width && t==height && src_bpp==num_bytes_per_pixel)){
    DPRINTF("s = %d, width %d\n",s,width);
    DPRINTF("t = %d, height %d\n",t,height);
    DPRINTF("src_bpp = %d, requested %d\n",src_bpp,num_bytes_per_pixel);
    return FsoiRequestWrongSizeErr;
  }

  memcpy( image_data, orig_data_ptr, s*t*num_bytes_per_pixel );

  return FsoiNoErr;
}

FsoiErr fsoi_ng_set_pos_ori(FsoiObj* theobj,
			    double x,double y,double z,
			    double qw,double qx,double qy,double qz) {
  DPRINTF("set pos & ori\n");
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;

  mfo->set_pos_ori(x,y,z,qw,qx,qy,qz);

  return FsoiNoErr;
}

FsoiErr fsoi_ng_get_width_height(FsoiObj* theobj,
				 int* width, int* height) {
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;
  mfo->get_width_height(width,height);
  return FsoiNoErr;
}

FsoiErr fsoi_ng_set_eyemap_face_colors( FsoiObj* theobj, int num, float* color_array, int n) {
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;

  osg::ref_ptr<osg::Vec4Array> cs = new osg::Vec4Array;
  int i=0;
  for (int i=0; i<n; i++) {
    cs->push_back( osg::Vec4( color_array[i*4], color_array[i*4+1], color_array[i*4+2], color_array[i*4+3]));
  }
  mfo->set_eyemap_face_colors( num, cs.get() );
  return FsoiNoErr;
}

FsoiErr fsoi_ng_set_eyemap_projection( FsoiObj* theobj, int num, float x1, float x2, float y1, float y2) {
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;
  mfo->setEyeMapProjection(num,x1,x2,y1,y2);
  return FsoiNoErr;
}

FsoiErr fsoi_ng_set_eyemap_geometry( FsoiObj* theobj, int num, float* vert_array, int n_verts,
				     unsigned char* fan_length_array, int n_fans ) {
  MyFsoiObj* mfo = (MyFsoiObj*)theobj->the_cpp_obj;

  osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array;
  for (int i=0; i<n_verts; i++) {
    osg::Vec3 nv = osg::Vec3( vert_array[i*2], vert_array[i*2+1], 0.0f );
    verts->push_back( nv );
  }

  osg::ref_ptr<osg::ByteArray> fan_lengths = new osg::ByteArray;
  for (int i=0; i<n_fans; i++) {
    fan_lengths->push_back( fan_length_array[i] );
  }

  mfo->setEyeMapGeometry(num, verts.get(),fan_lengths.get());

  return FsoiNoErr;
}

FsoiErr fsoi_ng_shutdown() {
  DPRINTF("shutdown\n");
  return FsoiNoErr;
}

} // closes: extern "C"

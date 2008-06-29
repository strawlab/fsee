#include "fsoi_ng.h"
#include <stdio.h>
#include <stdlib.h>

#define CHK(m)								\
if ((m) != FsoiNoErr) {						\
  fprintf(stderr,"%s:%d received unexpected fsoi error %d. Aborting\n",__FILE__,__LINE__,(m)); \
  exit((m));								\
}

int main( int argc, const char** argv ) {
  FsoiObj* fsoi_obj;

  unsigned char* image_data;
  int image_width, image_height, bpp;
  double x,y,z,qw,qx,qy,qz;
  float verts[12];
  unsigned char len;
  int count=0;
  float colors[4];
  
  CHK(fsoi_ng_init());

  CHK(fsoi_ng_new(&fsoi_obj,
		  "WT2/WT.osg",1000.0, // model name, scale
		  "brightday1_cubemap/", // skybox_basename
		  90.0,90.0, // FOV x, y
		  0.001,1e10, // near, far clip
		  64,64,"fb")); // texture width, height

#if 0
  CHK(fsoi_ng_run(fsoi_obj));
#else

  verts[0] = 0.0; verts[1] = 0.0; // x,y
  verts[2] = -5;  verts[3] = -5;
  verts[4] = -5;  verts[5] =  5;
  verts[6] = 5;   verts[7] =  5;
  verts[8] = 5;   verts[9] = -5;
  verts[10]= -5;  verts[11]= -5;
  
  len=6;
  CHK(fsoi_ng_set_eyemap_geometry( fsoi_obj, 0, &verts[0], 12, &len, 1 ));
  CHK(fsoi_ng_set_eyemap_geometry( fsoi_obj, 1, NULL, 0, NULL, 0 ));
  CHK(fsoi_ng_set_eyemap_projection( fsoi_obj, 0, -6*1.5,6*1.5, -6,6 ));

  x = 0.0;
  y = 0.0;
  z = 150.0;
  //z = 0.0;
  qw = 1.0;
  qx = 0.0;
  qy = 0.0;
  qz = 0.0;
  
  count=0; 
  while(count <300) {
    count++;
    CHK(fsoi_ng_set_pos_ori(fsoi_obj,x,y,z,qw,qx,qy,qz));
    CHK(fsoi_ng_render_frame(fsoi_obj,&image_data,&image_width,&image_height,&bpp));
    x = x+1.0;

    colors[0] = ((float)(count%100))/100.0f;
    colors[1] = 0;
    colors[2] = 0;
    colors[3] = 1.0;

    CHK(fsoi_ng_set_eyemap_face_colors(fsoi_obj,0,&colors[0],1));
  }
#endif

  CHK(fsoi_ng_delete(fsoi_obj));

  CHK(fsoi_ng_shutdown());
  
}

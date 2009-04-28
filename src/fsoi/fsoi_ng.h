/*
Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
Author: Andrew D. Straw
*/

#ifndef FSOI_NG_H
#define FSOI_NG_H

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef FSOI_NG_DLL
#ifdef FSOI_NG_EXPORTS
#define FSOI_NG_API __declspec(dllexport)
#else /* not defined: FSOI_NG_EXPORTS */
#define FSOI_NG_API __declspec(dllimport)
#endif /* FSOI_NG_EXPORTS */
#else /* not defined: FSOI_NG_DLL */
#define FSOI_NG_API extern
#endif /* FSOI_NG_DLL */


#ifdef __cplusplus
extern "C" {
#endif

typedef int FsoiErr;
typedef enum FsoiErrTypes
{
  FsoiNoErr=0,
  FsoiRenderFrameError,
  FsoiUnsupportedData,
  FsoiRequestWrongSizeErr,
  FsoiUnknownRenderImplementeation,
  FsoiAlreadyInitializedMemory,
  FsoiMemoryError,
  FsoiReadNodeFileError,
  FsoiNotImplementedError
}
FsoiErrTypes;


typedef struct FsoiObj FsoiObj;
struct FsoiObj {
  void* the_cpp_obj;
};

FSOI_NG_API FsoiErr fsoi_ng_init();
FSOI_NG_API FsoiErr fsoi_ng_shutdown();

FSOI_NG_API FsoiErr fsoi_ng_new(FsoiObj**,const char* filename, double scale, const char* skybox_basename,
				double im_xang, double im_yang, double near, double far,
				int im_width, int im_height, const char* render_implementation);
FSOI_NG_API FsoiErr fsoi_ng_delete(FsoiObj*);

FSOI_NG_API FsoiErr fsoi_ng_run(FsoiObj*);

FSOI_NG_API FsoiErr fsoi_ng_get_world_point(FsoiObj*,
                                            double* result_x,double* result_y,double*result_z,
                                            double* v1_x, double* v1_y, double* v1_z,
                                            double* v2_x, double* v2_y, double* v2_z);

FSOI_NG_API FsoiErr fsoi_ng_render_frame(FsoiObj* theobj,unsigned char** image_data_ptr, int* width, int* height, int* bytes_per_pixel);
FSOI_NG_API FsoiErr fsoi_ng_render_frame_copy(FsoiObj* theobj,unsigned char* image_data, int width, int height, int bytes_per_pixel);

FSOI_NG_API FsoiErr fsoi_ng_set_pos_ori(FsoiObj* fsoi_obj,
					double x,double y,double z,
					double qw,double qx,double qy,double qz);

FSOI_NG_API FsoiErr fsoi_ng_get_width_height(FsoiObj* fsoi_obj,
					     int* width, int* height);


FSOI_NG_API FsoiErr fsoi_ng_set_eyemap_face_colors( FsoiObj* theobj, int num, float* color_array, int n);

FSOI_NG_API FsoiErr fsoi_ng_set_eyemap_geometry( FsoiObj* theobj, int num, float* vert_array, int n_verts,
						 unsigned char* fan_length_array, int n_fans );

FSOI_NG_API FsoiErr fsoi_ng_set_eyemap_projection( FsoiObj* theobj, int num, float x1, float x2,
						   float y1, float y2);

#ifdef __cplusplus
}
#endif

#endif /* FSOI_NG_H */

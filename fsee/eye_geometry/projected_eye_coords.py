import fsee.plot_utils
from fsee.plot_utils import BasemapInstanceWrapper

class RapidPlotter(object):
    def __init__(self,optics=None):
        if optics == 'buchner71':

            biw_left = BasemapInstanceWrapper(optics = optics,
                                              proj = 'left_cyl',
                                              eye_name = 'left')
            biw_right = BasemapInstanceWrapper(optics = optics,
                                               proj = 'right_cyl',
                                               eye_name = 'right')
            biws = [biw_left,biw_right]
        elif optics == 'synthetic':
            biw_front = BasemapInstanceWrapper(proj='ortho',lon_0=0.0,
                                               optics=optics,
                                               eye_name = 'front')
            biw_back = BasemapInstanceWrapper(proj='ortho',lon_0=180.0,
                                              optics=optics,
                                              eye_name = 'back')

            biws = [biw_front,biw_back]
        else:
            raise NotImplementedError('')


        self.faces = {}
        self.slicer = {}

        self.tri_fans = {}

        self.eye_names = []

        for biw in biws:
            eye_name = biw.eye_name
            self.eye_names.append( eye_name )
            # Modified from fsee.plot_utils.plot_faces()
            if eye_name in biw.slicer:
                slicer_name = eye_name
            else:
                slicer_name = None

            self.slicer[eye_name] = biw.slicer[ slicer_name ]
            self.faces[eye_name] = biw.get_newfaces()[self.slicer[eye_name]]

            # New here
            self.tri_fans[eye_name] = []

            center_xs, center_ys = biw.get_rdirs2xy(slicer_name)
            for center_x,center_y,face in zip(center_xs,
                                              center_ys,
                                              self.faces[eye_name]):
                if face is not None:
                    face_x = [center_x] + list(face[0]) + [face[0][0]]
                    face_y = [center_y] + list(face[1]) + [face[1][0]]
                    tri_fan = face_x, face_y
                else:
                    tri_fan = None
                self.tri_fans[eye_name].append( tri_fan )

    def get_eye_names(self):
        return self.eye_names

    def get_faces(self,eye_name):
        return self.faces[eye_name]

    def get_tri_fans(self,eye_name):
        return self.tri_fans[eye_name]

    def get_slicer(self, eye_name):
        return self.slicer[eye_name]

    def flip_lon(self):
        return True

def show_eyes():
    rp = RapidPlotter(optics='buchner71')

    import pylab

    pylab.subplot(2,1,1)
    faces = rp.get_faces('left')
    for face in faces:
        if face is not None:
            pylab.plot(face[0],face[1],'b-')

    for tf in rp.get_tri_fans('left'):
        pylab.plot(tf[0],tf[1],'r-')

    pylab.subplot(2,1,2)
    faces = rp.get_faces('right')
    for face in faces:
        if face is not None:
            pylab.plot(face[0],face[1],'b-')

    pylab.show()

if __name__=='__main__':
    show_eyes()

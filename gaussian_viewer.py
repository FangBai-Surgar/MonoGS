
# https://github.com/isl-org/Open3D/blob/73508bcaba0a9a31e398bf8de76e3bbeaed81540/examples/python/visualization/video.py
import pickle
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import threading

import os

# from gui.gl_render import util as util
from gui.gl_render import util_lima as util
from gui.gl_render import util_gau_lima as util_gau
from gui.gl_render.render_ogl import OpenGLRenderer


from gui.gui_utils import create_frustum, cv_gl

import cv2

from OpenGL import GL as gl
import glfw


import json

import matplotlib.pyplot as plt

from gaussian_splatting.utils.graphics_utils import fov2focal, getWorld2View2

import torch
import rich



class CamInfo:
    def __init__(self, uid, R, T):
        self.uid = uid
        self.R = torch.from_numpy(R)
        self.T = torch.from_numpy(T)

    @property
    def camera_center(self):
        T = getWorld2View2(self.R, self.T).transpose(0, 1)
        return T.inverse()[3, :3]


def create_gaussians_gl (gaussians):
    gaussians_gl = util_gau.naive_gaussian()
    gaussians_gl.xyz = gaussians.get_xyz.detach().cpu().numpy()
    gaussians_gl.opacity = gaussians.get_opacity.detach().cpu().numpy()
    gaussians_gl.scale = gaussians.get_scaling.detach().cpu().numpy()
    gaussians_gl.rot = gaussians.get_rotation.detach().cpu().numpy()
    gaussians_gl.sh = gaussians.get_features.detach().cpu().numpy()[:, 0, :]
    return gaussians_gl






class Viewer:
    '''
        -- Mouse view control --
        Left button + drag         : Rotate.
        Ctrl + left button + drag  : Translate.
        Wheel button + drag        : Translate.
        Shift + left button + drag : Roll.
        Wheel                      : Zoom in/out.

        -- Keyboard view control --
        [/]          : Increase/decrease field of view.
        R            : Reset view point.
        Ctrl/Cmd + C : Copy current view status into the clipboard.
        Ctrl/Cmd + V : Paste view status from clipboard.

        -- General control --
        Q, Esc       : Exit window.
        H            : Print help message.
        P, PrtScn    : Take a screen capture.
        D            : Take a depth capture.
        O            : Take a capture of current rendering settings.   
    '''
    def __init__(self, viewpoint_stack = None, gaussians_gl = None, pose_vis_opts = 2, view_pose_W2C = None, width = None, height=None):        

        self.g_scale_modifier = 1.0
        self.camera_size = 0.1

        self.render_mode = -1 # -1,  4


        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.viewpoint_stack = viewpoint_stack
        self.gaussians_gl = gaussians_gl

        '''
            Open3D Visualizer Example
            link: https://github.com/isl-org/Open3D/blob/73508bcaba0a9a31e398bf8de76e3bbeaed81540/examples/python/visualization/video.py                 
        '''

        self.WIDTH, self.HEIGHT = 1200, 800
        if (width is not None) and (height is not None):
            self.WIDTH, self.HEIGHT = width, height

        self.window = gui.Application.instance.create_window ( "viewer: figure saved automatically", width=self.WIDTH, height=self.HEIGHT )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        # self.window.set_on_key(self._on_key)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)
        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "unlitLine"

        self.lit_geo = rendering.MaterialRecord()
        self.lit_geo.shader = "defaultUnlit"

        self.specular_geo = rendering.MaterialRecord()
        self.specular_geo.shader = "defaultLit"


        '''
            plot cameras a view furstums               
        '''
        if pose_vis_opts & 1:
            self.plot_trajectory(self.viewpoint_stack)

        '''
            plot the trajectory of camera centers               
        '''
        if pose_vis_opts & 2:
            self.plot_cameras(self.viewpoint_stack)


        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(45.0, bounds, bounds.get_center())
        self.widget3d.scene.show_axes(False)

        self.widget3d_width_ratio = 1.0
        self.widget3d_width = self.window.size.width * self.widget3d_width_ratio


        # set the position and orientation of the camera in case the view is user defined
        if view_pose_W2C is not None:
            C2W = np.linalg.inv(view_pose_W2C)
            frustum = create_frustum( C2W )
            viewpoint = frustum.view_dir
            self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])



        """
        For visualize 3DGS ellipsoids
        """
        self.g_camera = None
        self.window_gl  = None
        self.g_renderer = None
        self.render_img = None

        # a thread that helps to update view-control
        self.is_done = False
        threading.Thread(target=self._update_thread).start()

        app.run()



    def init_glfw(self):
        window_name = ""
        if not glfw.init():
            exit(1)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)        
        glfw.window_hint(glfw.HOVERED, glfw.FALSE)
        # glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(
            self.WIDTH, self.HEIGHT, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        glfw.set_window_opacity(window, 0.8)
        glfw.set_window_pos(window, -2000, -2000)
        return window



    def plot_cameras(self, viewpoint_stack = None, color = [0, 1, 0], camera_size=0.1):
        if (viewpoint_stack is None):
            return
        for camera in viewpoint_stack:
            name = "cam{}_".format(camera.uid)
            self.add_camera(camera, name, color=color, size=camera_size)


    def plot_trajectory(self, viewpoint_stack = None, colors = None):
        if (viewpoint_stack is None):
            return        
        (points, lines) = self.create_trajectory_lineset(viewpoint_stack)
        if colors is None:
            colors = [ [0, 0, 1]  for i in range(len(lines))]

        odometry_line_set = o3d.geometry.LineSet()
        odometry_line_set.points = o3d.utility.Vector3dVector(points)
        odometry_line_set.lines = o3d.utility.Vector2iVector(lines)
        odometry_line_set.colors = o3d.utility.Vector3dVector(colors)  # line colors, not point colors

        self.widget3d.scene.add_geometry("trajectory", odometry_line_set, self.lit)


    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 1.0
        self.widget3d_width = int(
            self.window.size.width * self.widget3d_width_ratio
        ) 
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )


    def _on_close(self):
        self.is_done = True       
        return True  # False would cancel the close
    

    # def _on_key(self, e):
    #     if e.key == gui.KeyName.S or e.key == gui.KeyName.s:
    #         self.save_figure()
    #     return True



    def _update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update the scene or any part of the UI.

        # create glfw context at update thread
        self.g_camera = util.Camera(h=self.HEIGHT, w=self.WIDTH)
        self.window_gl  = self.init_glfw()
        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)


        # use another offscreen buffer
        # fbo = gl.glGenFramebuffers(1)
        # render_buf = gl.glGenRenderbuffers(1)
        # gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, render_buf)
        # gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGB, self.WIDTH, self.HEIGHT)
        # gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fbo)
        # gl.glFramebufferRenderbuffer(gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, render_buf)
        # gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo)

        while not self.is_done:
            
            time.sleep(0.015)

            # compute_Gaussian_background here (update thread)
            if not glfw.window_should_close(self.window_gl):                
                (W2C, FoVx, FoVy, fx, fy, cx, cy, H, W) = self.get_current_cam()                
                self.render_img = self.render_gaussian_background_by_opengl(W2C, FoVy, H, W)
                rich.print(f"\ncurrent view info:\n\tWIDTH = {W}\n\tHEIGHT = {H}\n\tW2C = np.{repr(W2C)}")

            # Update the images. This must be done on the UI thread.
            def update():
                self.widget3d.scene.set_background([0, 0, 0, 1], self.render_img)                
                self.save_figure()
                time.sleep(0.001)

            if (not self.is_done): # need this to close properly. otherwise there will be "Segmentation fault (core dumped)"
                gui.Application.instance.post_to_main_thread(self.window, update)


        # use another offscreen buffer
        # gl.glDeleteFramebuffers(1,fbo)
        # gl.glDeleteRenderbuffers(1, render_buf)
        # gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)


        glfw.set_window_should_close(self.window_gl, glfw.TRUE)        
        glfw.destroy_window(self.window_gl)
        glfw.terminate()
        o3d.visualization.gui.Application.instance.quit()
        



    def save_figure(self):
        filename = "viewer_fig_automatic_save"
        height = self.window.size.height
        width = self.window.size.width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}.png", img)



    def add_camera(self, camera, name, color=[0, 0, 1], size=0.01):
        W2C = getWorld2View2(camera.R, camera.T)
        W2C = W2C.cpu().numpy()
        C2W = np.linalg.inv(W2C)
        frustum = create_frustum(C2W, color, size=size)
        self.widget3d.scene.add_geometry(name, frustum.line_set, self.lit)   
        self.widget3d.scene.set_geometry_transform(name, C2W.astype(np.float64))
        self.widget3d.scene.show_geometry(name, True)
        return frustum


    def create_trajectory_lineset(self, viewpoint_stack):
        camera_centers = []
        for viewpoint in viewpoint_stack:
            camera_centers.append ( viewpoint.camera_center.detach().cpu().numpy() )
        points = np.array( camera_centers )

        lines = []
        for i in range(len(camera_centers)-1):
            lines.append( [i, i+1] )
        
        return (points, lines)

    @staticmethod
    def vfov_to_hfov(vfov_deg, height, width):
        # http://paulbourke.net/miscellaneous/lens/
        return np.rad2deg(
            2 * np.arctan(width * np.tan(np.deg2rad(vfov_deg) / 2) / height)
        )

    def get_current_cam(self):
        w2c = cv_gl @ self.widget3d.scene.camera.get_view_matrix()
        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )
        vfov_deg = self.widget3d.scene.camera.get_field_of_view()
        hfov_deg = self.vfov_to_hfov(vfov_deg, image_gui.shape[1], image_gui.shape[2])
        FoVx = np.deg2rad(hfov_deg)
        FoVy = np.deg2rad(vfov_deg)
        fx = fov2focal(FoVx, image_gui.shape[2])
        fy = fov2focal(FoVy, image_gui.shape[1])
        cx = image_gui.shape[2] // 2
        cy = image_gui.shape[1] // 2
        H=image_gui.shape[1]
        W=image_gui.shape[2]
        return (w2c, FoVx, FoVy, fx, fy, cx, cy, H, W)



    def render_gaussian_background_by_opengl(self, W2C, FoVy, HEIGHT, WIDTH):
        glfw.poll_events()
        gl.glClearColor(0.0, 0.0, 0.0, 1.0) # black
        # gl.glClearColor(1.0, 1.0, 1.0, 1.0) # white
        gl.glClear(
            gl.GL_COLOR_BUFFER_BIT
            | gl.GL_DEPTH_BUFFER_BIT
            | gl.GL_STENCIL_BUFFER_BIT
        )
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)

        w = int(self.window.size.width * self.widget3d_width_ratio)
        h = int(self.window.size.height)

        # print(f"w = {w}, h = {h}")
        glfw.set_window_size(self.window_gl, w, h)

        self.g_camera.fovy = FoVy
        self.g_camera.update_resolution(h, w)

        # frustum = create_frustum(
        #     np.linalg.inv(cv_gl @ self.widget3d.scene.camera.get_view_matrix())
        # )

        C2W = np.linalg.inv(W2C)
        frustum = create_frustum( C2W )

        self.g_camera.position = frustum.eye.astype(np.float32)
        self.g_camera.target = frustum.center.astype(np.float32)
        self.g_camera.up = frustum.up.astype(np.float32)

        self.update_activated_renderer_state(self.gaussians_gl)
        
        self.g_renderer.sort_and_update(self.g_camera)
        width, height = glfw.get_framebuffer_size(self.window_gl)
        gl.glViewport(0, 0, width, height) # Viewport decided by buffer size, not window size. They're not always the same, tested on Mac wher buffer size is twice larger than frame size
        # print(f"frame buffer size: width = {width}, height = {height}")
        # gl.glScissor(0, 0, width, height)

        self.g_renderer.draw()

        '''
            https://stackoverflow.com/questions/12157646/how-to-render-offscreen-on-opengl
        '''
        #### gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0) # additional fbo buffer
        bufferdata = gl.glReadPixels(
            0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
        img = ( img * 0.5 ).astype(np.uint8) # alpha = 0.5
        img = cv2.flip(img, 0)

        # print(f"self.rendered_imgage: {img.shape}")
        self.render_img = o3d.geometry.Image(img)

        # glfw.swap_buffers(self.window_gl)
        # gl.glFinish()
        time.sleep(0.001)

        return self.render_img
        
        


    def update_activated_renderer_state(self, gaus):
        self.g_renderer.update_gaussian_data(gaus)
        self.g_renderer.sort_and_update(self.g_camera)
        self.g_renderer.set_scale_modifier(self.g_scale_modifier)
        self.g_renderer.set_render_mod(self.render_mode-3) # original 7-3
        self.g_renderer.update_camera_pose(self.g_camera)
        self.g_renderer.update_camera_intrin(self.g_camera)
        self.g_renderer.set_render_reso(self.g_camera.w, self.g_camera.h)



def read_camera_json (json_file_path):
    """
    {"id": 0, "img_name": "IMG_6292", "width": 1332, "height": 876, "position": [-1.4759880629577484, 1.6090724813669521, -2.7727036587765035], "rotation": [[0.5408209248789425, -0.8404510054983934, -0.03398285699915072], [0.003746154845639685, 0.042807333797130434, -0.999076322658611], [0.8411294154510098, 0.540194075800467, 0.026299561462536303]], "fy": 1034.9718637370904, "fx": 1035.4965990500061}
    """
    cam_infos = []

    with open(json_file_path, 'r') as json_file:
        contents = json.load(json_file)


    for cam_info in contents:
        uid = cam_info["id"]
        img_name = cam_info["img_name"]
        W = cam_info["width"]
        H = cam_info["height"]
        T = cam_info["position"]
        R = cam_info["rotation"]
        fx = cam_info["fx"]
        fy = cam_info["fy"]

        # get the world-to-camera transform and set R, T
        R = np.array(R)
        T = np.array(T)

        gR = np.transpose(R)
        gT = - np.transpose(R) @ T

        cam = CamInfo(
            uid, gR, gT
        )
        cam_infos.append(cam)

    return cam_infos


def load_replica_poses(path):
        cam_infos = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        poses = []
        for i in range(600):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            poses.append(pose)
            gR = np.transpose(pose[:3, :3])
            gT = - np.transpose(pose[:3, :3]) @ pose[:3, 3]

            cam = CamInfo(
                i, gR, gT
            )
            cam_infos.append(cam)

        return cam_infos

def gaussian_model_to_gaussian_data(gaussian_model):
    xyz = gaussian_model.get_xyz.cpu().numpy() 
    opacity = gaussian_model.get_opacity.cpu().numpy()
    scale = gaussian_model.get_scaling.cpu().numpy()
    rot = gaussian_model.get_rotation.cpu().numpy()
    features_dc = gaussian_model.get_features.cpu().numpy() #output with features_cd and features_rest
    
    # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    # for idx, attr_name in enumerate(extra_f_names):
    #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    # features_extra = np.transpose(features_extra, [0, 2, 1])
    extra_f_names = []
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                    features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    # sh = gaussian_model.get_features.detach().cpu().numpy()[:, 0, :]
    sh = gaussian_model.max_sh_degree
    print("len(shs): ", len(shs))
    print("len(xyz): ", len(xyz))
    print("len(rot): ", len(rot))
    print("len(scale): ", len(scale))
    print("len(opacity): ", len(opacity))
    print("len(features_dc): ", len(features_dc))
    # exit()  
    return util_gau.GaussianData(xyz, rot, scale, opacity, shs)

def main():

    camera_file_path = "/hdd/3DGS/bicycle/cameras.json"
    point_cloud_file_path = "/hdd/3DGS/bicycle/point_cloud/iteration_7000/point_cloud.ply"

    cam_infos = read_camera_json (camera_file_path)
    gaussians_gl = util_gau.load_ply(point_cloud_file_path)

    WIDTH = 1321
    HEIGHT = 818
    W2C = np.array([[ 0.98915207, -0.08082924, -0.12265864, -0.3948226 ],
    [ 0.08723773,  0.99504024,  0.04779924, -0.85472399],
    [ 0.1181867 , -0.05798113,  0.99129713,  3.59042859],
    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    

    Fig = Viewer(viewpoint_stack=cam_infos,  gaussians_gl= gaussians_gl,
                 width=WIDTH, height=HEIGHT, view_pose_W2C=W2C)




if __name__ == "__main__":
    main()



import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np

import cv2
import matplotlib




def render_mesh(input_mesh,
                pose : np.ndarray,
                fx : float, fy : float, cx : float, cy : float,
                width : int, height: int) -> tuple[np.ndarray, np.ndarray] :
    """
    open3d.visualization.rendering.OffscreenRenderer
    https://www.open3d.org/docs/latest/python_api/open3d.visualization.rendering.OffscreenRenderer.html

        Parameters:
            input_mesh (): 
                1) an open3d triangle mesh object : open3d.geometry.TriangleMesh or 
                2) a mesh file

        Returns:
            color_image (np.ndarray): shape(H, W, 3)
            depth_image (np.ndarray): shape(H, W)
    """

    if isinstance(input_mesh, str):
        mesh = o3d.io.read_triangle_mesh(input_mesh)
    elif isinstance(input_mesh, o3d.geometry.TriangleMesh):
        mesh = input_mesh
    else:
        raise TypeError("Unsupoorted argument for input_mesh type: ", type(input_mesh), "Expected input: an open3d mesh object or a mesh file.")

    o3d_renderer = rendering.OffscreenRenderer(width, height)
    o3d_renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
    o3d_renderer.scene.view.set_post_processing(False)

    # material for TriangleMesh (The base color does not replace the mesh's own colors.)
    material = rendering.MaterialRecord()
    material.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    material.shader = "defaultUnlit"

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    # add geometry
    o3d_renderer.scene.clear_geometry()
    o3d_renderer.scene.add_geometry("MyMeshModel", mesh, material, True)

    # setup camera. Sets camera view using bounding box of current geometry if the near_clip and far_clip parameters are not set
    o3d_renderer.setup_camera(intrinsic_matrix = intrinsic, extrinsic_matrix = pose, intrinsic_width_px = width, intrinsic_height_px = height)

    # render rgb and depth image
    rgb_img = o3d_renderer.render_to_image()
    depth_img = o3d_renderer.render_to_depth_image()

    return np.asarray(rgb_img), np.asarray(depth_img)






def depth2image (depth, colormap=None):
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    if colormap is not None:
        cmap = matplotlib.colormaps.get_cmap(colormap)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return depth



if __name__ == "__main__":

    # download test mesh
    armadillo_mesh = o3d.data.ArmadilloMesh()
    bunny = o3d.data.BunnyMesh()
    knot_mesh = o3d.data.KnotMesh()

    # use one of the downlaoded mesh
    mesh_file_path_name = armadillo_mesh.path

    # read mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file_path_name)

    # paint mesh with some color to make it look better, as the original is plain
    mesh.paint_uniform_color([1, 0.706, 0])
    
    # visualize mesh to find a proper camara for rendering
    if False:
        # Use the mouse to navigate and find the position which you want to render and then hit the button p
        # This will save a JSON file in the same directory called something like ScreenCamera_<somedate>.json, where you can read camera parameters
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.run()
        # to use the camera in rendering, follow the format:
        # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-10-03-16-23-45.json")
        # o3d_renderer.setup_camera(parameters.intrinsic, parameters.extrinsic)


    # instrinsic / calibration
    W, H = 1400, 750
    fx, fy, cx, cy = (650, 650, int((W+1)/2), int((H+1)/2))

    pose = np.array([
        [0.36738127116417013,
		-0.8819864755306871,
		-0.29517936679035572,
		0.0,],
		[-0.67195035829121907,
		-0.47113237696821203,
		0.57141666003415137,
		0.0,],
		[-0.64305032275094987,
		-0.011581897652776257,
		-0.76573633977803346,
		0.0,],
		[14.392634748282532,
		10.091394343402053,
		171.22850707427983,
		1.0] ]
    ).transpose()

    print(f"pose :\n{pose}")


    #### RENDER DATA

    print(render_mesh.__doc__)

    # use mesh file path
    (rgb_img, depth_img) = render_mesh (mesh_file_path_name, pose, fx, fy, cx, cy, width = W, height = H)

    # use o3d mesh object (has self-added color)
    (rgb_img, depth_img) = render_mesh (mesh, pose, fx, fy, cx, cy, width = W, height = H)

    
    if rgb_img is not None and depth_img is not None:
        
        color = cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGRA)
        depth = depth2image(depth_img)

        cv2.imshow("Color", color)    
        cv2.imshow("Depth", depth)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



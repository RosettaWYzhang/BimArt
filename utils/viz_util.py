import numpy as np
from collections.abc import Sequence
import scenepic as sp
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mplcolors
import matplotlib.cm as cm


def build_scenepic_motion(
    # object related
    obj_faces: np.ndarray = None,
    obj_verts_gt: np.ndarray = None,
    obj_parts: np.ndarray = None,
    obj_gt_states: np.ndarray = None, # for visualizing coordinate systems # [ph, 7]
    gt_obj_top_color: np.ndarray = sp.Colors.Purple,
    gt_obj_bottom_color: np.ndarray = sp.Colors.Lavender,
    # pred hand
    left_hand_vertices: np.ndarray = None,
    right_hand_vertices: np.ndarray = None,
    # pred hand dirvec
    left_hand_vertices_dirvec: np.ndarray = None,
    right_hand_vertices_dirvec: np.ndarray = None,
    left_hand_faces: np.ndarray = None,
    right_hand_faces: np.ndarray = None,
    left_hand_color: np.ndarray = sp.Colors.Yellow,
    right_hand_color: np.ndarray = sp.Colors.Pink,
    # contact scalar
    contact_scalar: np.ndarray = None,
    pred_left_contact: np.ndarray = None,
    pred_right_contact: np.ndarray = None,
    norm_high: float = 0.05,
) -> sp.Scene:
    scene = sp.Scene()
    width, height = 640, 640
    canvas_kwargs = dict(
        width=width,
        height=height,
        shading=build_shading(),
        ui_parameters=build_ui_parameters(),
    )

    ground_plane = scene.create_mesh(
        layer_id='Ground Plane',
        shared_color=sp.Colors.Beige,
    )
    ground_plane.add_quad(add_wireframe=True,
                          p0 = np.array([-2, -2, 0]),
                          p1 = np.array([2, -2, 0]),
                          p2 = np.array([2, 2, 0]),
                          p3 = np.array([-2, 2, 0]))
    ground_plane.add_coordinate_axes(length=0.05, thickness=0.02)    
    num_frames = len(right_hand_vertices)

    gt_obj_mesh = None
    if obj_verts_gt is not None:
        obj_colors_gt = np.zeros((len(obj_verts_gt[0]), 3))
        obj_colors_gt[obj_parts == 0] = gt_obj_top_color
        obj_colors_gt[obj_parts == 1] = gt_obj_bottom_color
    
        gt_obj_mesh = scene.create_mesh(
            layer_id='Object GT', 
        )
        gt_obj_mesh.add_mesh_without_normals(
            obj_verts_gt[0],
            obj_faces,
            colors=obj_colors_gt
        )

        left_contact_mesh = scene.create_mesh(
            layer_id='Pred Contact Left Hand', 
        )

        left_contact_mesh.add_mesh_without_normals(
            obj_verts_gt[0],
            obj_faces,
            colors=obj_colors_gt
        )

        right_contact_mesh = scene.create_mesh(
            layer_id='Pred Contact Right Hand', 
        )
        right_contact_mesh.add_mesh_without_normals(
            obj_verts_gt[0],
            obj_faces,
            colors=obj_colors_gt
        )

        focus_point = obj_verts_gt[0].mean(axis=0)
        camera = sp.Camera(
            center=focus_point + np.array([0, 0, 1]),
            look_at=focus_point,
            fov_y_degrees=25,
        )

        transform = np.eye(4)
        transform[:3, :3] = R.from_rotvec(obj_gt_states[0, 1:4]).as_matrix()
        transform[:3, 3] = obj_gt_states[0, 4:7]
        coord_mesh = scene.create_mesh(layer_id='Coordinate Frame GT')
        coord_mesh.add_coordinate_axes(length=0.05, thickness=0.02, transform=transform)
        coord_mesh.enable_instancing(obj_gt_states[0, 4:7])

    left_hand_mesh = None
    if left_hand_vertices is not None:
        left_hand_mesh = scene.create_mesh(
            layer_id='Left hand Prediction', shared_color=left_hand_color
        )
    
        left_hand_mesh.add_mesh_without_normals(
            (left_hand_vertices[0]), # need to be numpy!
            left_hand_faces,
        )

    right_hand_mesh = None
    if right_hand_vertices is not None:
        right_hand_mesh = scene.create_mesh(
            layer_id='Right Hand Prediction', shared_color=right_hand_color
        )
        right_hand_mesh.add_mesh_without_normals(
            (right_hand_vertices[0]), right_hand_faces
        )

    left_hand_mesh_dirvec = None
    if left_hand_vertices_dirvec is not None:
        left_hand_mesh_dirvec = scene.create_mesh(
            layer_id='Left hand Prediction Dirvec', shared_color=sp.Colors.Green, 
        )
        left_hand_mesh_dirvec.add_mesh_without_normals(
            (left_hand_vertices_dirvec[0]),
            left_hand_faces,
        )

    right_hand_mesh_dirvec = None
    if right_hand_vertices_dirvec is not None:
        right_hand_mesh_dirvec = scene.create_mesh(
            layer_id='Right hand Prediction Dirvec', shared_color=sp.Colors.Green, 
        )
        right_hand_mesh_dirvec.add_mesh_without_normals(
            (right_hand_vertices_dirvec[0]),
            right_hand_faces,
        )

    canvas_pred = scene.create_canvas_3d(camera=camera, **canvas_kwargs)
    canvas_sep_contact = scene.create_canvas_3d(camera=camera, **canvas_kwargs)

    for i in range(num_frames):
        if np.count_nonzero(obj_gt_states) != 0:
            transform = np.eye(4)
            transform[:3, :3] = R.from_rotvec(obj_gt_states[i, 1:4]).as_matrix()
            transform[:3, 3] = obj_gt_states[i, 4:7]

            coord_mesh_gt = scene.create_mesh(layer_id='Coordinate Frame GT')
            coord_mesh_gt.add_coordinate_axes(length=0.05, thickness=0.02, transform=transform)

            meshes_for_frame_pred = [coord_mesh_gt, ground_plane]
            meshes_for_frame_sep_contact = [coord_mesh_gt, ground_plane]
        else:

            meshes_for_frame_pred = [ground_plane]
            meshes_for_frame_sep_contact = [ground_plane]

        if left_hand_vertices is not None:
            left_hand_update = scene.update_mesh_positions(left_hand_mesh.mesh_id, left_hand_vertices[i])
            meshes_for_frame_pred.extend([left_hand_update])

        if right_hand_vertices is not None:
            right_hand_update = scene.update_mesh_positions(right_hand_mesh.mesh_id, right_hand_vertices[i])
            meshes_for_frame_pred.extend([right_hand_update])

        if left_hand_vertices_dirvec is not None:
            left_hand_dirvec_update = scene.update_mesh_positions(left_hand_mesh_dirvec.mesh_id, left_hand_vertices_dirvec[i])
            meshes_for_frame_pred.extend([left_hand_dirvec_update])
        if right_hand_vertices_dirvec is not None:
            right_hand_dirvec_update = scene.update_mesh_positions(right_hand_mesh_dirvec.mesh_id, right_hand_vertices_dirvec[i])
            meshes_for_frame_pred.extend([right_hand_dirvec_update])

        
        norm = mplcolors.Normalize(vmin=0, vmax=norm_high)
        if obj_verts_gt is not None:  
            if contact_scalar is not None:     
                contact_color = cm.jet_r(norm(contact_scalar[i]))[:, :3]
     
                obj_mesh_contact_update = scene.update_instanced_mesh(base_mesh_id=gt_obj_mesh.mesh_id, 
                                                                         positions=obj_verts_gt[i], 
                                                                         colors=contact_color)

                meshes_for_frame_sep_contact.extend([obj_mesh_contact_update])
                meshes_for_frame_pred.extend([obj_mesh_contact_update])


            else:
                gt_obj_mesh_update = scene.update_mesh_positions(gt_obj_mesh.mesh_id, obj_verts_gt[i])
                meshes_for_frame_pred.extend([gt_obj_mesh_update])

            if pred_right_contact is not None:
                pred_left_color = cm.jet_r(norm(pred_left_contact[i]))[:, :3]
                pred_right_color = cm.jet_r(norm(pred_right_contact[i]))[:, :3]

    
                pred_left_obj_mesh_contact_update = scene.update_instanced_mesh(base_mesh_id=left_contact_mesh.mesh_id, 
                                                                                positions=obj_verts_gt[i], 
                                                                                colors=pred_left_color)
                pred_right_obj_mesh_contact_update = scene.update_instanced_mesh(base_mesh_id=right_contact_mesh.mesh_id, 
                                                                                positions=obj_verts_gt[i], 
                                                                                colors=pred_right_color)        

                meshes_for_frame_sep_contact.extend([pred_left_obj_mesh_contact_update, pred_right_obj_mesh_contact_update])
    
        _ = canvas_pred.create_frame(meshes=meshes_for_frame_pred,
                                focus_point=focus_point,
                                frame_id="World Space Frame Prediction %d" % (i))
 

        _ = canvas_sep_contact.create_frame(meshes=meshes_for_frame_sep_contact,
                        focus_point=focus_point,
                        frame_id="World Space Frame Contact Prediction %d" % (i))

    canvas_pred.set_layer_settings({})
    canvas_sep_contact.set_layer_settings({})
    return scene


def build_shading(
    ambient_light_color: Sequence[float] = (0.5, 0.5, 0.5),
    directional_light_color: Sequence[float] = (0.5, 0.5, 0.5),
    bg_color: Sequence[float] = (1, 1, 1),
    directional_light_dir: Sequence[float] = (1, 1, 2),
) -> sp.Shading:
    """Lighting setup for a scenepic scene."""
    return sp.Shading(
        ambient_light_color=ambient_light_color,
        directional_light_color=directional_light_color,
        directional_light_dir=directional_light_dir,
        bg_color=bg_color,
    )


def build_ui_parameters(
    mouse_wheel_translation_speed: float = 1.0e-03,
    pointer_rotation_speed: float = 1.0e-02,
) -> sp.UIParameters:
    """Creates UI options object for ScenePic scene."""
    return sp.UIParameters(
        pointer_rotation_speed=pointer_rotation_speed,
        mouse_wheel_translation_speed=mouse_wheel_translation_speed,
    )

def swatches(color_bar=None, norm_high: float = 0.05) -> str:
  """Creates a palette using the colors and their names.

  Args:
    colors: An array or a sequence of 3-tuple of floating point values that
      represent colors.
    names: The name for each color.
    width: The width of each color block.

  Returns:
    A HTML string with the colors and their corresponding names.
  """
  str_list = []
  # Add label for the color bar
  str_list.append('<div style="font-weight: bold;">Contact Map (min: 0; max: %s)</div>' %str(norm_high))
  scalar_values = np.linspace(color_bar[0], color_bar[1], 100)  # Adjust the number of values as needed
  norm = mplcolors.Normalize(vmin=color_bar[0], vmax=color_bar[1])
  color_bar_colors = cm.jet_r(norm(scalar_values))

  color_bar_template = (
    '<div style="display: flex; flex-direction: row;">'
    '<div style="width: 100%; height: 20px; background: linear-gradient(to right, {color_gradient});"></div>'
    '</div>'
  )

  # Create a linear gradient for the color bar
  color_gradient = ', '.join([mplcolors.to_hex(color[:3]) for color in color_bar_colors])

  str_list.append(color_bar_template.format(color_gradient=color_gradient))
  return '<br>\n'.join(str_list)
# You get blocks of colors with strings, which you can use as a legend.
# _Color here is a numpy array with shape (N, 3)
# You can use this so that you don't have to explain every time what each color means on each ScenePic.

# '<span style="color: {color}">{block}</span>: {name:30s}'`
# `block = width * chr(9608) `


def get_legend(norm_high: float = 0.05) -> str:
    colorbar = np.array([0, norm_high])
    return swatches(color_bar=colorbar, norm_high=norm_high)
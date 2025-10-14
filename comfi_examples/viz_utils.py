import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
import meshcat
import time


def meshcat_material(r, g, b, a):
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material


def meshcat_transform(x, y, z, q, u, a, t):
    return np.array(pin.XYZQUATToSE3([x, y, z, q, u, a, t]))


# === Function to place a marker ===
def place(viewer, name, pos):
    T = tf.translation_matrix(pos)
    viewer[f"world/{name}"].set_transform(T)


def add_sphere(vis, path, radius=0.01, color=0x0000FF):
    """Add a colored sphere to Meshcat."""
    vis[path].set_object(
        g.Sphere(radius), g.MeshPhongMaterial(color=color, shininess=100)
    )


def addViewerSphere(viz, name, size, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(
            meshcat.geometry.Sphere(size), meshcat_material(*rgba)
        )
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addSphere(name, size, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def applyViewerConfiguration(viz, name, xyzquat):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_transform(meshcat_transform(*xyzquat))
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.applyConfiguration(name, xyzquat)
        viz.viewer.gui.refresh()
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def add_markers_to_meshcat(
    viewer,
    mks_dict,
    marker_names=None,
    radius=0.01,
    default_color=0xFF5500,
    opacity=0.9,
    per_marker_color=None,
):
    """
    Create one sphere per marker under /markers/<name> in the Meshcat tree.
    - mks_dict[name] is expected to be shape (T, 3) in **meters**.
    - per_marker_color: optional dict name -> 0xRRGGBB
    """
    if marker_names is None:
        marker_names = list(mks_dict.keys())

    sphere = g.Sphere(radius)  # one geometry object reused for all markers
    for name in marker_names:
        color = (per_marker_color or {}).get(name, default_color)
        mat = g.MeshLambertMaterial(
            color=color, opacity=opacity, transparent=(opacity < 1.0)
        )
        node = viewer["markers"][name]
        node.set_object(sphere, mat)
        # start hidden until first valid frame
        node.set_property("visible", False)


def set_markers_frame(viewer, markers, t, marker_names=None, unit_scale=1.0):
    """
    Accepts either:
      A) dict name -> (T,3) array
      B) list length T of dicts name -> (3,)
    """
    is_list_of_dicts = isinstance(markers, list)
    if marker_names is None:
        if is_list_of_dicts:
            names = set()
            for d in markers:
                names.update(d.keys())
            marker_names = sorted(names)
        else:
            marker_names = list(markers.keys())

    for name in marker_names:
        if is_list_of_dicts:
            d = markers[t]
            if name not in d:
                viewer["markers"][name].set_property("visible", False)
                continue
            P = np.asarray(d[name], dtype=float).reshape(
                3,
            )
        else:
            P = np.asarray(markers[name][t], dtype=float).reshape(
                3,
            )

        if not np.isfinite(P).all():
            viewer["markers"][name].set_property("visible", False)
            continue

        P = P * unit_scale
        viewer["markers"][name].set_property("visible", True)
        viewer["markers"][name].set_transform(tf.translation_matrix(P))


def addViewerBox(viz, name, sizex, sizey, sizez, rgba):
    if isinstance(viz, pin.visualize.MeshcatVisualizer):
        viz.viewer[name].set_object(
            meshcat.geometry.Box([sizex, sizey, sizez]), meshcat_material(*rgba)
        )
    elif isinstance(viz, pin.visualize.GepettoVisualizer):
        viz.viewer.gui.addBox(name, sizex, sizey, sizez, rgba)
    else:
        raise AttributeError("Viewer %s is not supported." % viz.__class__)


def box_between_frames(
    viz, name, T_c0, T_c2, thickness=0.02, height=0.02, rgba=(0.2, 0.7, 0.9, 0.8)
):
    """
    Draw a rectangular box connecting the origins of frames c0 and c2.
    The box's local X axis is aligned with the segment (p2 - p1).

    viz   : Pinocchio MeshcatVisualizer
    name  : meshcat node name (string)
    T_c0  : 4x4 world->c0 transform
    T_c2  : 4x4 world->c2 transform
    thickness, height : box cross-section (meters)
    rgba  : color with alpha
    """
    p1 = np.asarray(T_c0[:3, 3], dtype=float)
    p2 = np.asarray(T_c2[:3, 3], dtype=float)
    v = p2 - p1
    L = float(np.linalg.norm(v))
    if L < 1e-12:
        # Degenerate: frames coincide. Draw a tiny dot-sized box.
        L = 1e-6
        ex = np.array([1.0, 0.0, 0.0])
        ey = np.array([0.0, 1.0, 0.0])
        ez = np.array([0.0, 0.0, 1.0])
    else:
        ex = v / L  # direction along the segment
        up = np.array([0.0, 0.0, 1.0])
        if abs(ex @ up) > 0.95:  # avoid near-parallel 'up'
            up = np.array([0.0, 1.0, 0.0])
        ey = np.cross(up, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)

    # Rotation with columns = object axes in world (X along segment)
    R = np.column_stack((ex, ey, ez))
    c = 0.5 * (p1 + p2)

    # Build world transform for the box
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c
    T[2, 3] -= 0.1
    # Create geometry (length along local X)
    L += 0.1
    addViewerBox(viz, name, L, thickness, height, rgba)

    # Place it
    viz.viewer[name].set_transform(T)


def set_tf(viz, name, T_world_obj):
    viz.viewer[name].set_transform(T_world_obj)


def draw_table(viz, T_world_table):
    """
    Build a simple table (top + 4 legs) and place it at T_world_table.
    Uses addViewerBox() for both MeshCat and Gepetto viewers.

    Parameters
    ----------
    viz : pin.visualize.MeshcatVisualizer or GepettoVisualizer
    T_world_table : (4,4) ndarray homogeneous transform (world_T_table)
    """
    # ---- table parameters (meters) ----
    L, W, T = 0.90, 1.80, 0.04  # tabletop length, width, thickness
    H = 0.95  # total height
    LEG = 0.05  # leg square cross-section
    INSET = 0.05  # leg inset from edges

    # ---- names (unique in the scene) ----
    top_name = "table_top"
    leg_names = [
        "table_leg_00",
        "table_leg_01",
        "table_leg_10",
        "table_leg_11",
    ]

    # ---- create geometries (via your helper) ----
    # tabletop (brown)
    addViewerBox(viz, top_name, L, W, T, rgba=[0.80, 0.60, 0.40, 1.0])
    # legs (slightly darker)
    for n in leg_names:
        addViewerBox(viz, n, LEG, LEG, H - T, rgba=[0.45, 0.45, 0.45, 1.0])

    # ---- local poses (relative to the table frame) ----
    def homog(R=np.eye(3), t=(0, 0, 0)):
        Tm = np.eye(4)
        Tm[:3, :3] = R
        Tm[:3, 3] = np.array(t, dtype=float)
        return Tm

    # tabletop center is at z = H - T/2
    T_local_top = homog(t=(0.0, 0.0, H - T / 2))

    # leg centers
    xs = [+L / 2 - INSET - LEG / 2, -L / 2 + INSET + LEG / 2]
    ys = [+W / 2 - INSET - LEG / 2, -W / 2 + INSET + LEG / 2]
    z_leg = (H - T) / 2
    T_local_legs = [
        homog(t=(xs[0], ys[0], z_leg)),
        homog(t=(xs[0], ys[1], z_leg)),
        homog(t=(xs[1], ys[0], z_leg)),
        homog(t=(xs[1], ys[1], z_leg)),
    ]

    # ---- apply world transforms ----
    set_tf(viz, top_name, T_world_table @ T_local_top)
    for n, Tl in zip(leg_names, T_local_legs):
        set_tf(viz, n, T_world_table @ Tl)


def make_visuals_gray(visual_model: pin.GeometryModel, rgba=(0.4, 0.4, 0.4, 0.7)):
    for go in visual_model.geometryObjects:
        go.overrideMaterial = True
        go.meshColor = np.array([*rgba], dtype=float)


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with
    vec2.
    """
    a, b = (
        (vec1 / np.linalg.norm(vec1)).reshape(3),
        (vec2 / np.linalg.norm(vec2)).reshape(3),
    )
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def display_force_meshcat(viz, phi, M_se3, name="arrow"):
    """
    Version avec flèche partant du dessus de la plateforme
    """
    import meshcat.geometry as g
    import meshcat.transformations as tf

    M_se3_temp = M_se3.copy()
    # color = 0x0000ff
    color = 0xFF8000
    radius = 0.01

    phi_transformed = phi.se3Action(M_se3)
    force = phi_transformed.linear
    length = np.linalg.norm(force) * 1e-3

    if length < 1e-6:
        return

    force_direction = force / np.linalg.norm(force)

    platform_thickness = 0.01
    start_position = M_se3.translation.copy()
    start_position[2] += platform_thickness + 0.005

    arrow_center = start_position + force_direction * length * 0.5

    meshcat_default_axis = np.array([0, 1, 0])
    Rot = rotation_matrix_from_vectors(meshcat_default_axis, force)

    M_se3_temp.translation = arrow_center
    M_se3_temp.rotation = M_se3.rotation @ Rot

    arrow_geom = g.Cylinder(length, radius)
    viz.viewer[name].set_object(arrow_geom, g.MeshLambertMaterial(color=color))

    transform = tf.compose_matrix(
        translate=M_se3_temp.translation,
        angles=tf.euler_from_matrix(M_se3_temp.rotation),
    )
    viz.viewer[name].set_transform(transform)


def animate(
    scene,
    jcp,
    jcp_names,
    jcp_hpe,
    jcp_names_hpe,
    q_ref,
    q_robot,
    force_data,
    forceplates_dims_and_centers,
    sync,
    freq,
    step=5,
    i0=0,
):
    unit_scale = 1.0
    viewer = scene.viewer
    fp_dims, fp_centers = forceplates_dims_and_centers
    # jcp from mocap
    add_markers_to_meshcat(
        viewer,
        jcp,
        marker_names=jcp_names,
        radius=0.020,
        default_color=0xFF0000,
        opacity=0.95,
    )

    # jcp from hpe
    if jcp_hpe is not None:
        add_markers_to_meshcat(
            viewer,
            jcp_hpe,
            marker_names=jcp_names_hpe,
            radius=0.025,
            default_color=0x00FF00,
            opacity=0.95,
        )

    # Mapping capteurs -> plateformes
    sensor_mapping = {1: 0, 2: 1, 3: 2}

    for i in range(i0, len(q_ref), step):
        # draw JCP spheres
        set_markers_frame(viewer, jcp, i, marker_names=jcp_names, unit_scale=unit_scale)

        # set_markers_frame(viewer, mks_dict, i, marker_names=mks_names, unit_scale=unit_scale)

        jcp_hpe_renamed = []
        if jcp_hpe is not None:
            for d in jcp_hpe:
                jcp_hpe_renamed.append({k + "_hpe": v for k, v in d.items()})
            set_markers_frame(
                viewer,
                jcp_hpe_renamed,
                i,
                marker_names=jcp_names_hpe,
                unit_scale=unit_scale,
            )

        if sync is not None:
            cam_idx = sync["cam_idx"]
            # robot frames exist in q_robot indices [0 .. len-1]
            rr = i - cam_idx
            if 0 <= rr < len(q_robot):
                scene.viz_robot.display(q_robot[rr, :])

        scene.viz_human.display(q_ref[i, :])

        if force_data is not None:
            for sensor_id in force_data.keys():
                arrow_name = f"force_sensor{sensor_id}"
                scene.viz_human.viewer[arrow_name].delete()

            for sensor_id, data in force_data.items():
                if i < len(data["Fx"]) and sensor_id in sensor_mapping:
                    plate_idx = sensor_mapping[sensor_id]
                    if plate_idx < len(fp_centers):
                        # Données de la frame i
                        fx = data["Fx"][i]
                        fy = data["Fy"][i]
                        fz = -1 * (data["Fz"][i])
                        mx = data["Mx"][i]
                        my = data["My"][i]
                        mz = data["Mz"][i]

                        if not np.any(np.isnan([fx, fy, fz])):
                            phi = pin.Force(
                                np.array([fx, fy, fz]), np.array([mx, my, mz]) * 1e-3
                            )

                            M_se3 = pin.SE3.Identity()
                            M_se3.translation = np.array(fp_centers[plate_idx])
                            M_se3.translation[2] += 0.02

                            display_force_meshcat(
                                scene.viz_human, phi, M_se3, f"force_sensor{sensor_id}"
                            )
        time.sleep(0.90 * 1 / freq)

    #     images.append(viewer.get_image())

    # imageio.mimsave("video.mp4", images, fps=40)


# def animate(scene,
#             mks_dict,
#             mks_names,
#             q_ref: np.ndarray,
#             q_robot: np.ndarray,
#             jcp: np.ndarray,
#             sync,
#             step: int = 5,
#             i0: int = 0):
#     # markers
#     # add_markers_to_meshcat(scene.viz_human.viewer, mks_dict, marker_names=mks_names, radius=0.025, default_color=0x2E86DE, opacity=0.95)
#     unit_scale = 1.0

#     # loop
#     for i in range(i0, len(q_ref), step):
#         # draw JCP spheres
#         for j in range(jcp.shape[1]):
#             sphere_name = f'jcp_mocap{j}'
#             addViewerSphere(scene.viz_human, sphere_name, 0.025, [0, 1, 0, 1.0])
#             applyViewerConfiguration(scene.viz_human, sphere_name, np.hstack((jcp[i, j, :], np.array([0, 0, 0, 1]))))

#         # set_markers_frame(scene.viz_human.viewer, mks_dict, i, marker_names=mks_names, unit_scale=unit_scale)

#         if sync is not None:
#             cam_idx = sync["cam_idx"]
#             # robot frames exist in q_robot indices [0 .. len-1]
#             rr = i - cam_idx
#             if 0 <= rr < len(q_robot):
#                 scene.viz_robot.display(q_robot[rr, :])

#         scene.viz_human.display(q_ref[i, :])

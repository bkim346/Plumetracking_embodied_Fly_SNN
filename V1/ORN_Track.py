import numpy as np
import h5py
import os
from phi.torch import flow
from tqdm import trange
from typing import Tuple
from pathlib import Path
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from phi import vis
from pathlib import Path
from scipy.interpolate import interp1d

from flygym import Fly, SingleFlySimulation, Camera
from flygym.examples.olfaction import PlumeNavigationTask
from flygym.examples.olfaction.plume_tracking_arena import OdorPlumeArena
from flygym.examples.locomotion import PreprogrammedSteps

from common import WalkingState

from enum import Enum
import cv2
from flygym.util import get_data_path

from snn_controller import SNNPlumeController

# Create an output directory relative to the current script location
output_dir = Path(__file__).parent / "outputs" 

# Create the directories if they don't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Output will be saved to: {output_dir}")




# Set camera/simulation view
main_camera_name = "birdeye_camera"
arena = OdorPlumeArena(
    output_dir / "plume_tcropped.hdf5", main_camera_name=main_camera_name,
    plume_simulation_fps=8000, dimension_scale_factor=0.25
)

# Plume marker
inflow_pos = (4, 30)
inflow_radius = 1
inflow_scaler = 0.2
velocity_grid_size = 0.5
smoke_grid_size = 0.25


# get the angle of the vector in world coordinates
def get_vector_angle(v):
    return np.arctan2(v[1], v[0])


# change an array to a set of probabilities (sum to 1)
# this is used to bias crosswind walking
def to_probability(x):
    # the difference between the two values reflects
    # the probability of each entry
    x += np.abs(np.min(x)) + 1
    return x / np.sum(x)

def get_debug_str(
    accumulated_evidence, curr_angle, target_angle, crosswind_success_proba
):
    """
    Get a string that represents the state of the controller
    """
    crosswind_success_proba_str = " ".join(
        [f"{co:.2f}" for co in crosswind_success_proba]
    )
    return [
        f"Accumulated evidence: {accumulated_evidence:.2f}",
        f"Fly orientation: {np.rad2deg(curr_angle):.2f}",
        f"Target angle: {np.rad2deg(target_angle):.2f}",
        f"Crosswind success proba: {crosswind_success_proba_str}",
    ]


def get_walking_icons():
    """
    Get all icons representing the walking directions
    """
    icons_dir = get_data_path("flygym", "data") / "etc/locomotion_icons"
    icons = {}
    for key in ["forward", "left", "right", "stop"]:
        icon_path = icons_dir / f"{key}.png"
        icons[key] = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
    return {
        WalkingState.FORWARD: icons["forward"],
        WalkingState.TURN_LEFT: icons["left"],
        WalkingState.TURN_RIGHT: icons["right"],
        WalkingState.STOP: icons["stop"],
    }


def get_inflow_circle(inflow_pos, inflow_radius, camera_matrix):
    """
    Compute the xy locations of the inflow circle in the camera view
    """
    # draw a circle around the inflow position (get x y pos of
    # a few points on the circle)
    circle_x, circle_y = [], []
    for angle in np.linspace(0, 2 * np.pi + 0.01, num=50):
        circle_x.append(inflow_pos[0] + inflow_radius * np.cos(angle))
        circle_y.append(inflow_pos[1] + inflow_radius * np.sin(angle))

    xyz_global = np.array([circle_x, circle_y, np.zeros_like(circle_x)])

    # project those points on the camera view
    # Camera matrices multiply homogenous [x, y, z, 1] vectors.
    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    # Project world coordinates into pixel space. See:
    # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
    xs, ys, s = camera_matrix @ corners_homogeneous

    # x and y are in the pixel coordinate system.
    x = np.rint(xs / s).astype(int)
    y = np.rint(ys / s).astype(int)

    return x, y


def render_overlay(
    rendered_img,
    accumulated_evidence,
    fly_orientation,
    target_angle,
    crosswind_success_proba,
    icon,
    window_size,
    inflow_x,
    inflow_y,
):
    """
    Helper function to modify the simulation rendered images
    """

    if rendered_img is not None:
        sub_strings = get_debug_str(
            accumulated_evidence,
            get_vector_angle(fly_orientation),
            target_angle,
            crosswind_success_proba,
        )
        # put string at the top left corner of the image
        for j, sub_string in enumerate(sub_strings):
            rendered_img = cv2.putText(
                rendered_img,
                sub_string,
                (5, window_size[1] - (len(sub_strings) - j + 1) * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        # put the icon just under the debug string
        rendered_img[
            window_size[1] - 100 - icon.shape[1] : window_size[1] - 100,
            0 : icon.shape[1],
            :,
        ] = icon

        # draw the inflow circle as a free line
        rendered_img = cv2.polylines(
            rendered_img,
            [np.array([list(zip(inflow_x, inflow_y))])],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
        )

    return rendered_img


def is_close_to_boundary(pos, arena_size, margin=5.0):
    """
    Check if the fly is close to the boundary

    """
    return (
        pos[0] < margin
        or pos[0] > arena_size[0] - margin
        or pos[1] < margin
        or pos[1] > arena_size[1] - margin
    )



from dm_control.mujoco import Camera as DmCamera

# write the same loop as before but with the new controller
timestep = 1e-4
run_time = 10.0

np.random.seed(777)
arena = OdorPlumeArena(
    output_dir / "plume_tcropped"
    ".hdf5",
    main_camera_name=main_camera_name,
    plume_simulation_fps=800,
    dimension_scale_factor=0.25,
    num_sensors=4,
)

# Define the fly
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

fly = Fly(
    enable_adhesion=True,
    draw_adhesion=True,
    enable_olfaction=True,
    enable_vision=False,
    contact_sensor_placements=contact_sensor_placements,
    # Here the opposite spawn position can be tried (65.0, 15.0, 0.25)
    spawn_pos=(65.0, 45.0, 0.25),
    spawn_orientation=(0, 0, -np.pi),
)

wind_dir = [1.0, 0.0]
ctrl = SNNPlumeController(timestep)

cam_params = {"mode":"fixed",
    "pos": (
                0.50 * arena.arena_size[0],
                0.15 * arena.arena_size[1],
                1.00 * arena.arena_size[1],
            ),
    "euler":(np.deg2rad(15), 0, 0), "fovy":60}

cam = Camera(
    attachment_point=arena.root_element.worldbody,
    camera_name=main_camera_name,
    timestamp_text = False,
    camera_parameters=cam_params
)
sim = PlumeNavigationTask(
    fly=fly,
    arena=arena,
    cameras=[cam],
)

sim.reset(0)
dm_cam = DmCamera(
    sim.physics,
    camera_id=cam.camera_id,
    width=cam.window_size[0],
    height=cam.window_size[1],
)
camera_matrix = dm_cam.matrix
arena_inflow_pos = np.array(inflow_pos) / arena.dimension_scale_factor * smoke_grid_size
target_inflow_radius = 5.0
inflow_x, inflow_y = get_inflow_circle(
    arena_inflow_pos,
    target_inflow_radius,
    camera_matrix,
)

walking_icons = get_walking_icons()

obs, info = sim.reset(0)

for i in trange(np.rint(run_time / timestep).astype(int)):
    fly_orientation = obs["fly_orientation"][:2]
    fly_orientation /= np.linalg.norm(fly_orientation)
    close_to_boundary = is_close_to_boundary(obs["fly"][0][:2], arena.arena_size)
    dn_drive = ctrl.step(obs["odor_intensity"])

    obs, reward, terminated, truncated, info = sim.step(dn_drive)

    icon = walking_icons[ctrl.curr_state][:, :, :3]
    rendered_img = sim.render()[0]
    rendered_img = render_overlay(
        rendered_img,
        0.0,
        fly_orientation,
        0.0,
        [0.5,0.5],
        icon,
        cam.window_size,
        inflow_x,
        inflow_y,
    )
    obs_list = []
    if rendered_img is not None:
        cam._frames[-1] = rendered_img

    if np.linalg.norm(obs["fly"][0][:2] - arena_inflow_pos) < target_inflow_radius:
        print("The fly reached the inflow")
        break
    elif truncated:
        print("The fly went out of bound")
        break

    obs_list.append(obs)


cam.save_video(output_dir / "plume_navigation_controller.mp4")
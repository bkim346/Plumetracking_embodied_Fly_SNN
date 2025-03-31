# generate_dataset.py
# Updated to use SimplePlumeNavigationController for behavior generation

import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import trange
from enum import Enum

# Constants
PLUME_PATH = "outputs/plume.hdf5"
SAVE_PATH = "outputs/train_data.npz"
ARENA_SIZE = (80, 60)
NUM_SEQUENCES = 500
SEQUENCE_LENGTH = 300
TIMESTEP = 0.01
PLUME_SIM_FPS = 200
DIMENSION_SCALE = 0.25
INTENSITY_SCALE = 1.0

BEHAVIORS = ["walk", "stop", "turn_left", "turn_right"]

# --------------------------------------------------
# Enum for behavior states
class WalkingState(Enum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3

# --------------------------------------------------
# Utility functions

def get_vector_angle(v):
    return np.arctan2(v[1], v[0])

def to_probability(x):
    x += np.abs(np.min(x)) + 1
    return x / np.sum(x)

# --------------------------------------------------
# Load plume

def load_plume():
    with h5py.File(PLUME_PATH, "r") as f:
        plume = f["plume"][:]
    return plume  # shape: [T, H, W]

# --------------------------------------------------
# Simple controller based on Demir et al. behavior

class SimplePlumeNavigationController:
    def __init__(self, timestep, wind_dir=[-1.0, 0.0], seed=0):
        self.timestep = timestep
        self.wind_dir = wind_dir
        np.random.seed(seed)

        self.state_map = {
            WalkingState.FORWARD: 0,
            WalkingState.STOP: 1,
            WalkingState.TURN_LEFT: 2,
            WalkingState.TURN_RIGHT: 3,
        }

        self.accumulated_evidence = 0.0
        self.accumulation_decay = 0.0001
        self.accumulation_odor_gain = 0.05
        self.accumulation_threshold = 20.0

        self.default_decision_interval = 0.75  # s
        self.since_last_decision_time = 0.0
        self.min_evidence = -1 * self.accumulation_decay * self.default_decision_interval / timestep
        self.dn_drive_update_interval = 0.1
        self.dn_drive_update_steps = int(self.dn_drive_update_interval / self.timestep)

        self.curr_state = WalkingState.STOP
        self.target_angle = np.nan
        self.to_upwind_angle = np.nan
        self.upwind_success = [0, 0]

    def get_target_angle(self):
        up_wind_angle = get_vector_angle(self.wind_dir) - np.pi
        to_upwind_angle = np.tanh(self.accumulated_evidence) * np.pi / 4 - np.pi / 4
        crosswind_success_proba = to_probability(self.upwind_success)
        to_upwind_angle *= np.random.choice([-1, 1], p=crosswind_success_proba)
        target_angle = up_wind_angle + to_upwind_angle
        if target_angle > np.pi:
            target_angle -= 2 * np.pi
        elif target_angle < -np.pi:
            target_angle += 2 * np.pi
        return target_angle, to_upwind_angle

    def angle_to_behavior(self, fly_orientation):
        fly_angle = get_vector_angle(fly_orientation)
        angle_diff = self.target_angle - fly_angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        if np.isnan(self.target_angle):
            return WalkingState.STOP
        elif angle_diff > np.deg2rad(10):
            return WalkingState.TURN_LEFT
        elif angle_diff < -np.deg2rad(10):
            return WalkingState.TURN_RIGHT
        else:
            return WalkingState.FORWARD

    def step(self, odor_intensity, fly_orientation, curr_time):
        if (
            self.accumulated_evidence > self.accumulation_threshold
            or self.since_last_decision_time > self.default_decision_interval
        ):
            if self.accumulated_evidence > self.accumulation_threshold:
                self.upwind_success = [0, 0]

            if self.to_upwind_angle < np.deg2rad(-45):
                self.upwind_success[0] += (
                    1 if self.accumulated_evidence > self.min_evidence else -1
                )
            elif self.to_upwind_angle > np.deg2rad(45):
                self.upwind_success[1] += (
                    1 if self.accumulated_evidence > self.min_evidence else -1
                )

            self.target_angle, self.to_upwind_angle = self.get_target_angle()
            self.accumulated_evidence = 0.0
            self.since_last_decision_time = 0.0
        else:
            self.accumulated_evidence += odor_intensity * self.accumulation_odor_gain - self.accumulation_decay

        if (
            np.rint(curr_time / self.timestep) % self.dn_drive_update_steps == 0
        ):
            self.curr_state = self.angle_to_behavior(fly_orientation)

        self.since_last_decision_time += self.timestep
        return self.state_map[self.curr_state]

# --------------------------------------------------
# Simulate fly path and label behavior

def simulate_fly_path(plume, speed=0.1):
    T, H, W = plume.shape
    pos = np.array([np.random.uniform(2, 10), np.random.uniform(20, 40)])
    angle = np.random.uniform(0, 2 * np.pi)
    path = []

    for t in range(SEQUENCE_LENGTH):
        dx = np.cos(angle) * speed
        dy = np.sin(angle) * speed
        pos += np.array([dx, dy])
        pos = np.clip(pos, [0, 0], [W - 1, H - 1])
        angle += np.random.normal(0, 0.1)
        # Convert physical position in mm → plume grid indices
        x_idx = int(pos[0] / DIMENSION_SCALE)
        y_idx = int(pos[1] / DIMENSION_SCALE)

        if 0 <= x_idx < plume.shape[2] and 0 <= y_idx < plume.shape[1]:
            odor = plume[t % T, y_idx, x_idx] * INTENSITY_SCALE
        else:
            odor = 0.0  # Outside bounds → no odor

        path.append((pos.copy(), odor, angle))

    return path

# --------------------------------------------------
# Generate dataset

def generate_training_data():
    plume = load_plume()
    X, y = [], []
    controller = SimplePlumeNavigationController(timestep=TIMESTEP)

    for _ in trange(NUM_SEQUENCES):
        controller.__init__(TIMESTEP)  # reset state
        path = simulate_fly_path(plume)

        for t, (pos, odor, angle) in enumerate(path):
            fly_orientation = np.array([np.cos(angle), np.sin(angle)])
            behavior = controller.step(
                odor_intensity=odor,
                fly_orientation=fly_orientation,
                curr_time=t * TIMESTEP
            )
            X.append([odor, odor])
            y.append(behavior)

    X = np.stack(X).astype(np.float32)
    y = np.array(y).astype(np.int64)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savez(SAVE_PATH, X=X, y=y)
    print(f"Saved dataset to {SAVE_PATH}")

if __name__ == "__main__":
    generate_training_data()

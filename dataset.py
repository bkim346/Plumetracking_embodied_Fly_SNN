import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class FlyPlumeDataset(torch.utils.data.Dataset):
    def __init__(self, df, timesteps=112, shuffle_trajs=False):
        self.timesteps = timesteps
        self.samples = []

        if shuffle_trajs:
            grouped = list(df.groupby("trjnum"))
            np.random.shuffle(grouped)
            df = pd.concat([g for _, g in grouped]).reset_index(drop=True)

        # Group by fly trajectory
        for trj_id, group in df.groupby("trjnum"):
            odor = group[["odor_left1", "odor_left2", "odor_right1", "odor_right2"]].values.astype("float32")
            labels = group["label"].values.astype("int64")

            # Skip short trajectories
            if len(odor) < timesteps:
                continue

            # Create rolling windows
            for i in range(len(odor) - timesteps):
                input_seq = odor[i:i+timesteps]  # shape [timesteps, 4]
                label = labels[i+timesteps]      # label at the *end* of the window
                self.samples.append((input_seq, label, trj_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, fly_id = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y), fly_id
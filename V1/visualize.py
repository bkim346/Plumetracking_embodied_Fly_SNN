import h5py
import numpy as np
import matplotlib.pyplot as plt

# Open the file
with h5py.File("v1/outputs/plume.hdf5", "r") as f:
    print("Keys:", list(f.keys()))
    plume = f["plume"][:]  # Shape: (num_frames, height, width)
    print("Shape:", plume.shape)

    # Plot a single frame
    plt.imshow(plume[0], cmap='inferno')
    plt.title("Plume Frame 0")
    plt.colorbar(label="Odor intensity")
    plt.show()

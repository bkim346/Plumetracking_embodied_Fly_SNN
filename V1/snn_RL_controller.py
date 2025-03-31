import torch
import numpy as np
import norse.torch as norse
import torch.nn as nn

class SNNcontroller:
    def __init__(self, timestep=1e-4):
        self.device - torch.device("cuda")
        self.timestep = timestep

        self.orn_layer = nn.Linear(2,100) # 2 antennas to 100 ORNs
        self.orn_lif = norse.LIFCell()

        self.decision_layer = nn.Linear(100,4) # 100 ORNs to 4 behavior outputs
        self.decision_lif = norse.LIFCell()

        self.states = (None,None)


        # Mappings each decision neuron to a dn drive
        self.dn_map = {
            0: np.array([1.0, 1.0]),    # Walk
            1: np.array([0.0, 0.0]),    # Stop
            2: np.array([-0.4, 1.2]),   # Turn left
            3: np.array([1.2, -0.4]),   # Turn right
        }


    def step(self, odor_intensity, fly_oreintation, close_to _boundary, curr_time):

        x = torch.tensor(odor_intensity,dtype=torch.float32).unsqueeze(0)


        # Go through the SNN

        x1 = self.orn_layer(x)
        z1,s1 = self.orn_lif(x1, self.states[0])
        x2 = self.decision_layer(z1)
        z2, s2 = self.decision_lif(x2, self.states[1])
        self.states = (s1, s2)

        # decode behavior

        decision = torch.argmax(z2).item()
        dn_drive = self.dn_map[decision]

        return dn_drive
    

    def reset(self):
        self.states = (None,None)
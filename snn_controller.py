import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils
from enum import Enum
import numpy as np

from common import WalkingState

dn_drives = {
    WalkingState.FORWARD: np.array([1.0, 1.0]),
    WalkingState.TURN_LEFT: np.array([-0.4, 1.2]),
    WalkingState.TURN_RIGHT: np.array([1.2, -0.4]),
    WalkingState.STOP: np.array([0.0, 0.0]),
}

class LearnableThresholdLIF(nn.Module):
    def __init__(self, beta=0.9, spike_grad=None, init_threshold=1.0):
        super().__init__()
        self.beta = beta
        self.spike_grad = spike_grad if spike_grad else snn.surrogate.fast_sigmoid()
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.lif = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)

    def forward(self, x, mem=None):
        # Temporarily override the threshold during the forward pass
        self.lif.threshold = self.threshold
        return self.lif(x, mem)


# Trainable SNN Model
class PlumetrackingSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = 0.9
        self.timesteps = 112  # Number of timesteps to simulate
        self.spike_grad = surrogate.fast_sigmoid()
        
        # Layer definitions
        self.fc1 = nn.Linear(4, 1300)
        self.dropout1 = nn.Dropout(p=0.2)
        self.norm1 = nn.LayerNorm(1300)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.bias1 = nn.Parameter(torch.zeros(1300))  # Trainable neuron-specific bias

        self.fc2 = nn.Linear(1300, 340)
        self.dropout2 = nn.Dropout(p=0.2)
        self.norm2 = nn.LayerNorm(340)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.bias2 = nn.Parameter(torch.zeros(340))

        self.fc3 = nn.Linear(340, 1400)
        self.norm3 = nn.LayerNorm(1400)
        self.lif3 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad)
        self.bias3 = nn.Parameter(torch.zeros(1400))

        self.fc4 = nn.Linear(1400, 4)
        self.norm4 = nn.LayerNorm(4)
        self.lif4 =  LearnableThresholdLIF(beta=self.beta, spike_grad=self.spike_grad, init_threshold=0.5)
        self.bias4 = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        snn.utils.reset(self)
        x = x.to(next(self.parameters()).device)

        # Initialize accumulator for the membrane potential of the final layer
        mem4_sum = 0.0
        spk4_all = []

        for t in range(x.shape[1]):
            xt = x[:, t, :]  # shape: [batch_size, 4]
            # Normalize odor inputs (values range ~0–300, most < 20)
            xt = xt/50 + 1e-3

            x1 = self.fc1(xt) + self.bias1
            spk1, _ = self.lif1(x1)

            x2 = self.fc2(spk1)+self.bias2
            spk2, _ = self.lif2(x2)

            x3 = self.fc3(spk2)+self.bias3
            spk3, _ = self.lif3(x3)

            x4 = self.fc4(spk3) +self.bias4
            spk4, mem4 = self.lif4(x4)

            spk4_all.append(spk4)
            mem4_sum += mem4

        spk4_stack = torch.stack(spk4_all, dim=1)
        spike_counts = spk4_stack.sum(dim=1)


        # Average the membrane potentials over timesteps
        mem4_avg = mem4_sum / x.shape[1]

        return spike_counts, spk4_stack

    


# Hardcoded SNN Controller
class SNNPlumeController:
    def __init__(self, model, timestep, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.curr_state = WalkingState.STOP  # Default state
        self.reset()
    
    def step(self, odor_intensity):
        x = torch.tensor(odor_intensity, dtype=torch.float32).to(self.device)
        spk_out, _ = self.model(x)

        if torch.sum(spk_out) == 0:
            chosen_state = WalkingState.STOP
        else:
            chosen_idx = torch.argmax(spk_out).item()
            chosen_state = WalkingState(chosen_idx)

        self.curr_state = chosen_state
        return dn_drives[chosen_state]
    
    def reset(self):
        self.mem1 = self.mem2 = self.mem3 = self.mem4 = None
        

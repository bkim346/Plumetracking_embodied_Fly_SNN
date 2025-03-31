#  Embodied Drosophila Simulation

This is a project after https://github.com/bkim346/Flygym, where instead of a manually coded controller, I wanted to explore creating a controller that is architechtually constrained by the connectome, and where sensory processing is done with layers of spiking neural networks to emulate a more realistic fly. The behavior to emulate will be the smoke tracking we see in  Demir et al. 2020(https://elifesciences.org/articles/57524). 

It will start with a simple version where it is just a few layers based on the known anatomy of the fly olfactory processing system, to ideally emulating a wide population of neurons from sensory encoding at the antenna to the descending neruons going to the ventral nerve cord (VNC).

Version 1: Controller is a single layer SNN emulating antennal olfactory receptor neurons (ORNs).


--

##  Demo

Here’s a quick look at the fly in action:  
![Fly walking and tracking odor plume](./outputs/plume_tracking/plume_dataset/plume_tracking_fly.gif)

---

## Project Overview

- **What this is:** A beginner project based on the Flygym tutorial.  
- **What I did:** Ran the simulation, explored the code, and started understanding how physics engines like MuJoCo are used in these models.  
- **Why I did it:** To build a foundation for future work in embodied neural simulations and explore its feasibility for simulating realistic behavior driven by complex neural inputs.

---

## How to Run It Yourself

1. **Clone the repo:**

git clone https://github.com/bkim346/Flygym.git

cd Flygym

2. **Install the requirements:**

pip install -r requirements.txt

3. **Generate odor plume:**

python Simulate_odorplume.py

The plume file I generated is too large to upload it to github. The seed I used was 777 and is set in line 12.

4. **Run the simulation:**

python plume_tracking.py


---

## What I Learned

- How to run Flygym simulations and use their API.
- Basics of MuJoCo and how it handles physics in biomechanical models.
- How embodied simulations are structured and how neural controllers can be connected.

---

## What’s Next?

For now, this is a basic walkthrough of the tutorial. Moving forward, I’m planning to:

- Modify the neural controllers to experiment with different behaviors.
- Explore more complex environments and tasks for the fly to perform.

---

## Project Structure

- outputs/
  - plume_tracking/
    - plume_dataset/        
      - plume_tracking_fly.mp4   # Plume tracking video 
      - plume_tracking_fly.gif   # Plume tracking gif

- Simulate_odorplume.py        # Simulate odor plume
- plume_tracking.py            # Fly odor tracking simulation code
- requirements.txt
- README.md

Acknowledgments
Big thanks to the Flygym and NeuroMechFly teams for their awesome tools and documentation.

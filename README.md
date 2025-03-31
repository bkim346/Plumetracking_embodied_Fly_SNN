#  Embodied Drosophila Simulation

This is a project after https://github.com/bkim346/Flygym, where instead of a manually coded controller, I wanted to explore creating a controller that is architechtually constrained by the connectome, and where sensory processing is done with layers of spiking neural networks to emulate a more realistic fly. The behavior to emulate will be the smoke tracking we see in  Demir et al. 2020(https://elifesciences.org/articles/57524). 

It will start with a simple version where it is just a few layers based on the known anatomy of the fly olfactory processing system, to ideally emulating a wide population of neurons from sensory encoding at the antenna to the descending neruons going to the ventral nerve cord (VNC).

Version 1: 1300 ORNs to 340 AL PNs to 1400 LH neurons to 4 DN neurons. The connections are not contrained by connectome and is just a proof of concept to get the parameters correct for the SNN to learn how to track based on ~2700 imaged and tracked real life plume tracking flies.

Future stpes: Organize the OR to AL/ glomeruli connection based on the available receptor projection data I compiled and summarized in "Drosophila OR to glom.xlsx".

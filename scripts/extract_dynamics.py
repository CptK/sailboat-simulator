#!/usr/bin/env python3
"""Extract dynamics matrices from simulation for LQR controller.
                                                                                                                                
Run this script whenever the simulation model changes to get updated matrices.
Copy the printed output to controller/lqr.py in the get_lqr_controller() function.
"""                                                                                                                           
from simulation.sailboat_simulation import SailboatSimulation                                                                 


sim = SailboatSimulation()
sim.reset()
A, B, dt = sim.get_heading_dynamics()


print("Copy these values to controller/lqr.py:")
print(f"A_discrete = np.array({repr(A)})")
print(f"B_discrete = np.array({repr(B)})")
print(f"dt_estimated = {dt}")

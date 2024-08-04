import os
import sys
import numpy as np


time_str = sys.argv[1]

seed_number = int(input("Enter seed index: "))
os.chdir(os.path.dirname(__file__))
os.chdir("../cache")

waveform_A = np.load(f"{time_str}_record_waveform_A.npy")[-1]
waveform_B = np.load(f"{time_str}_record_waveform_B.npy")[-1]

os.chdir("../seeds")
np.save(f"seed_{seed_number}_A_{time_str}.npy", waveform_A)
np.save(f"seed_{seed_number}_B_{time_str}.npy", waveform_B)


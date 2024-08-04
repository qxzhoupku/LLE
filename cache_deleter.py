import os
import sys


time_str = sys.argv[1]

os.chdir(os.path.dirname(__file__))
os.chdir("../cache")

os.remove(f"{time_str}_record_power_A.npy")
os.remove(f"{time_str}_record_power_B.npy")
os.remove(f"{time_str}_record_waveform_A.npy")
os.remove(f"{time_str}_record_waveform_B.npy")
os.remove(f"{time_str}_zetas.npy")
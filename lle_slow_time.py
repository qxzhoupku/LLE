import numpy as np
from numba import jit, objmode
import matplotlib.pyplot as plt
import os, sys
import glob
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import cProfile
from parameters import iter_number, zeta_ini, zeta_end, zeta_step, f_A, f_B, J_back_r, delta_t, D_int, time_str, rng, cProfile_test, noise_flag, power_interval, noise_level # type: ignore

# import sys
# import time

# Change directory to output folder
current_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_path))
output_path = "../output"
if not os.path.exists(output_path):
    os.mkdir(output_path)
os.chdir(output_path)


@jit(nopython=True)
def noise(rng, noise_level = noise_level):
    white_noise = rng.standard_normal() + 1j * rng.standard_normal()
    return white_noise * noise_level
    # smooth_noise = gaussian_filter1d(white_noise, sigma=10)
    # return smooth_noise
    # # return np.random.random(mode_number) * np.exp(1j * np.random.random(mode_number)) / 2


@jit(nopython=True)
def split_step(A_0, zeta, f, D_int, delta_t, B, B_avg_pow, J_back_r=0, noise_flag=False, rng = rng):
    A_3 = np.exp((-1 + 1j * (-zeta + np.abs(A_0)**2 + 2 * B_avg_pow)) * delta_t) * A_0 + f * delta_t
    A_4 = A_3 + 1j * J_back_r * delta_t * B # backscattering term from backwards mode
    if noise_flag:
        A_4 += noise(rng)
    return A_4


def result_plot(record_power_A, record_power_B, zetas, time_str, f_A, f_B, J_back_r, zeta_ini, zeta_end):
    # Plot power
    plt.figure()
    plt.plot(zetas, record_power_A, label=f'Power A, f_A = {f_A}', alpha = 0.7)
    plt.plot(zetas, record_power_B, label=f'Power B, f_B = {f_B}', alpha = 0.7)
    plt.xlim(zeta_ini, zeta_end)
    plt.title(f"Power, J = {J_back_r}")
    plt.xlabel("detuning")
    plt.legend()
    plt.savefig(f"{time_str}_power.png", dpi=600)
    print("Power saved")
    plt.show()


################
# Main loop
@jit(nopython=True)
def main_loop(iter_number, zeta_ini, zeta_step, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, power_interval):
    zeta = zeta_ini - zeta_step
    # for i in tqdm(range(iter_number), desc="Processing"):
    for i in range(iter_number):
        zeta = zeta + zeta_step
        power_A = np.abs(A)**2
        power_B = np.abs(B)**2
        A_new = split_step(A, zeta, f_A, D_int, delta_t, B, power_B, J_back_r, noise_flag, rng)
        B_new = split_step(B, zeta, f_B, D_int, delta_t, A, power_A, J_back_r, noise_flag, rng)
        A, B = A_new, B_new

        if i % power_interval == 0:
            zeta_index = i // power_interval
            zetas[zeta_index] = zeta
            record_power_A[zeta_index] = np.abs(A)**2
            record_power_B[zeta_index] = np.abs(B)**2
################




# Initialization
A = noise(rng)
B = noise(rng)

zetas = np.zeros(iter_number // power_interval)
record_power_A = np.zeros(iter_number // power_interval)
record_power_B = np.zeros(iter_number // power_interval)



print("Start main loop")
if cProfile_test:
    cProfile.run("main_loop(iter_number, zeta_ini, zeta_step, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, power_interval)", f"{time_str}_profile.prof")
else:
    main_loop(iter_number, zeta_ini, zeta_step, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, power_interval)
print("End main loop")

plt.ioff()



os.chdir(os.path.dirname(__file__))
os.chdir("../cache")


print("length of zetas:", len(zetas))

os.chdir("../output")

result_plot(record_power_A, record_power_B, zetas, time_str, f_A, f_B, J_back_r, zeta_ini, zeta_end)


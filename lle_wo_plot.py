import numpy as np
from numba import jit, objmode
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import cProfile
from parameters import mode_number, iter_number, plot_interval, record_interval, zeta_ini, zeta_end, zetas, f_A, f_B, J_back_r, delta_t, D_int, time_str, rng, plot_flag, cProfile_test, noise_flag # type: ignore

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
def noise(mode_number, rng):
    white_noise = rng.standard_normal(mode_number) + 1j * rng.standard_normal(mode_number)
    return white_noise
    # smooth_noise = gaussian_filter1d(white_noise, sigma=10)
    # return smooth_noise
    # # return np.random.random(mode_number) * np.exp(1j * np.random.random(mode_number)) / 2


@jit(nopython=True)
def cal_power(x):
    mode_number = len(x)
    return np.sum(np.abs(x)**2) / mode_number


@jit(nopython=True)
def split_step(A_0, zeta, f, D_int, delta_t, B, B_avg_pow, J_back_r=0, noise_flag=False, rng = rng):
    A_1 = np.exp(1j * (np.abs(A_0)**2 + B_avg_pow) * delta_t) * A_0
    A_1_freq = np.fft.fft(A_1)
    A_2_freq = np.exp(-(1 + 1j * zeta + 1j * D_int) * delta_t) * A_1_freq
    A_2 = np.fft.ifft(A_2_freq)
    A_3 = A_2 + f * delta_t
    A_4 = A_3 + 1j * J_back_r * delta_t * B # backscattering term from backwards mode
    if noise_flag:
        A_4 += noise(mode_number, rng) * 0.0001
    return A_4


################
# Main loop
@jit(nopython=True)
def main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B):
    # for i in tqdm(range(iter_number), desc="Processing"):
    for i in range(iter_number):
        zeta = zetas[i]
        power_A = cal_power(A)
        power_B = cal_power(B)
        record_power_A[i] = power_A
        record_power_B[i] = power_B
        A_new = split_step(A, zeta, f_A, D_int, delta_t, B, power_B, J_back_r, noise_flag, rng)
        B_new = split_step(B, zeta, f_B, D_int, delta_t, A, power_A, J_back_r, noise_flag, rng)
        A, B = A_new, B_new

        if i % record_interval == 0:
            record_waveform_A[i // record_interval] = A
            record_waveform_B[i // record_interval] = B
################




# Initialization
A = noise(mode_number, rng) * 0.0001
B = noise(mode_number, rng) * 0.0001
A_freq = np.fft.fftshift(np.fft.fft(A))
B_freq = np.fft.fftshift(np.fft.fft(B))

record_power_A = np.zeros(iter_number)
record_power_B = np.zeros(iter_number)
record_waveform_A = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)
record_waveform_B = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)


print("Start main loop")
if cProfile_test:
    cProfile.run("main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B)", f"{time_str}_profile.prof")
else:
    main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B)
print("End main loop")

# store D_int
np.savetxt(f"{time_str}_D_int.txt", D_int)

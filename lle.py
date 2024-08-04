import numpy as np
from numba import jit, objmode
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import cProfile
from parameters import mode_number, iter_number, plot_interval, record_interval, zeta_ini, zeta_end, zetas, f_A, f_B, J_back_r, delta_t, D_int, time_str, rng, plot_flag, cProfile_test, noise_flag, seed_number # type: ignore

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

        if i % plot_interval == 0 and plot_flag == True:
            with objmode():
                figure_plot(A, B, i, zeta, ax, ax_freq, line_A, line_B, line_A_freq, line_B_freq)
################


def figure_plot(A, B, i, zeta, ax, ax_freq, line_A, line_B, line_A_freq, line_B_freq):
    line_A.set_ydata(np.abs(A))
    line_B.set_ydata(np.abs(B))
    y_max = np.max([np.max(np.abs(A)), np.max(line_A.get_ydata())])
    ax.set_ylim(0, 1.2 * y_max)
    ax.title.set_text(f"zeta = {zeta:.2f}, proc = {i / iter_number * 100:.2f}%, f_A = {f_A}, J = {J_back_r}")

    A_freq = np.fft.fftshift(np.fft.fft(A))
    B_freq = np.fft.fftshift(np.fft.fft(B))
    line_A_freq.set_ydata(np.abs(A_freq))
    line_B_freq.set_ydata(np.abs(B_freq))
    y_freq_max = np.max(np.abs(A_freq))
    ax_freq.set_ylim(np.min(np.abs(A_freq)), 1.2 * y_freq_max)
    ax_freq.set_yscale('log')

    line_A_phase.set_ydata(np.angle(A))
    line_B_phase.set_ydata(np.angle(B))

    fig.canvas.draw()
    fig.canvas.flush_events()




# Initialization
if seed_number != -1:
    # load seed
    file_seed_A = glob.glob(f"../seeds/seed_{seed_number}_A_*.npy")[0]
    file_seed_B = glob.glob(f"../seeds/seed_{seed_number}_B_*.npy")[0]
    A = np.load(file_seed_A)
    B = np.load(file_seed_B)
else:
    A = noise(mode_number, rng) * 0.0001
    B = noise(mode_number, rng) * 0.0001
A_freq = np.fft.fftshift(np.fft.fft(A))
B_freq = np.fft.fftshift(np.fft.fft(B))

record_power_A = np.zeros(iter_number)
record_power_B = np.zeros(iter_number)
record_waveform_A = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)
record_waveform_B = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)

if plot_flag:
    plt.ion()
    fig, axs = plt.subplots(4)
    fig.set_size_inches(5, 7)
    # fig.canvas.manager.window.resizable(True, True)
    ax, ax_freq, ax_phase_A, ax_phase_B = axs[0], axs[1], axs[2], axs[3]
    line_A, = ax.plot(np.abs(A), alpha = 0.7)
    line_B, = ax.plot(np.abs(B), alpha = 0.7)
    xs_freq = np.arange(-mode_number / 2, mode_number / 2)
    line_A_freq, = ax_freq.plot(xs_freq, np.abs(A_freq), alpha = 0.7)
    line_B_freq, = ax_freq.plot(xs_freq, np.abs(B_freq), alpha = 0.7)
    line_A_phase, = ax_phase_A.plot(np.angle(A), alpha = 0.7)
    line_B_phase, = ax_phase_B.plot(np.angle(B), alpha = 0.7)

print("Start main loop")
if cProfile_test:
    cProfile.run("main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B)", f"{time_str}_profile.prof")
else:
    main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B)
print("End main loop")

plt.ioff()

# store D_int
np.savetxt(f"{time_str}_D_int.txt", D_int)

# save results to cache
os.chdir("../cache")
np.save(f"{time_str}_record_power_A.npy", record_power_A)
np.save(f"{time_str}_record_power_B.npy", record_power_B)
np.save(f"{time_str}_record_waveform_A.npy", record_waveform_A)
np.save(f"{time_str}_record_waveform_B.npy", record_waveform_B)
np.save(f"{time_str}_zetas.npy", zetas)
np.save(f"{time_str}_parameters.npy", np.array([time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end], dtype=object))
os.chdir(output_path)
print("Results saved")


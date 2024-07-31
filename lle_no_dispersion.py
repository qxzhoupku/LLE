import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import cProfile
from parameters import mode_number, iter_number, plot_interval, record_interval, zeta_ini, zeta_end, zetas, f_A, f_B, J_back_r, delta_t, D_int, time_str, plot_flag, cProfile_test, noise_flag # type: ignore

# import sys
# import time

# Change directory to output folder
current_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_path))
output_path = "../output"
if not os.path.exists(output_path):
    os.mkdir(output_path)
os.chdir(output_path)

def noise(mode_number):
    white_noise = np.random.randn(mode_number)
    # white_noise = np.random.normal(scale=0.1, size=mode_number)
    return white_noise
    smooth_noise = gaussian_filter1d(white_noise, sigma=10)
    return smooth_noise
    # return np.random.random(mode_number) * np.exp(1j * np.random.random(mode_number)) / 2

# fft = lambda x: np.fft.fftshift(np.fft.fft(x))
# ifft = lambda x: np.fft.ifft(np.fft.ifftshift(x))

@jit(nopython=True)
def cal_power(x):
    mode_number = len(x)
    return np.sum(np.abs(x)**2) / mode_number

@jit(nopython=True)
def split_step(A_0, zeta, f, D_int, delta_t, B, J_back_r=0, noise_flag=False):
    B_avg_pow = cal_power(B)
    A_3 = np.exp((-(1 + J_back_r) + 1j * (-zeta + np.abs(A_0)**2 + B_avg_pow)) * delta_t) * A_0 + f * delta_t
    A_4 = A_3 + 1j * (1 - np.exp(-J_back_r * delta_t)) * B[::-1] # backscattering term from backwards mode
    if noise_flag:
        A_4 += noise(mode_number) * 0.0001
    return A_4

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
    fig.canvas.draw()
    fig.canvas.flush_events()







# Initialization
A = noise(mode_number)
B = noise(mode_number)
A_freq = np.fft.fftshift(np.fft.fft(A))
B_freq = np.fft.fftshift(np.fft.fft(B))

record_power_A = np.zeros(iter_number)
record_power_B = np.zeros(iter_number)
record_waveform_A = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)
record_waveform_B = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)

if plot_flag:
    plt.ion()
    fig, axs = plt.subplots(2)
    ax, ax_freq = axs[0], axs[1]
    line_A, = ax.plot(np.abs(A))
    line_B, = ax.plot(np.abs(B))
    xs_freq = np.arange(-mode_number / 2, mode_number / 2)
    line_A_freq, = ax_freq.plot(xs_freq, np.abs(A_freq))
    line_B_freq, = ax_freq.plot(xs_freq, np.abs(B_freq))

################
# Main loop
def main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag):
    for i in tqdm(range(iter_number), desc="Processing"):
        zeta = zetas[i]
        A_new = split_step(A, zeta, f_A, D_int, delta_t, B, J_back_r, noise_flag)
        B_new = split_step(B, zeta, f_B, D_int, delta_t, A, J_back_r, noise_flag)
        A, B = A_new, B_new
        record_power_A[i] = cal_power(A)
        record_power_B[i] = cal_power(B)

        if i % record_interval == 0:
            record_waveform_A[i // record_interval] = A
            record_waveform_B[i // record_interval] = B

        if i % plot_interval == 0 and plot_flag == True:
            figure_plot(A, B, i, zeta, ax, ax_freq, line_A, line_B, line_A_freq, line_B_freq)
################

print("Start main loop")
if cProfile_test:
    cProfile.run("main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag)", f"{time_str}_profile.prof")
else:
    main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag)
print("End main loop")

plt.ioff()

# store D_int
np.savetxt(f"{time_str}_D_int.txt", D_int)

# Plot power
plt.figure()
plt.plot(zetas, record_power_A, label='Power A')
plt.plot(zetas, record_power_B, label='Power B')
plt.xlim(zeta_ini, zeta_end)
plt.title("Power_no_dispersion")
plt.legend()
plt.savefig(f"{time_str}_power_no_dispersion.png", dpi=600)

# Plot waveform heatmap
record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)
record_freq_A = record_freq_A.T
record_freq_B = record_freq_B.T
record_waveform_A = record_waveform_A.T
record_waveform_B = record_waveform_B.T

plt.figure()
plt.imshow(np.abs(record_waveform_A), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Waveform_A_no_dispersion")
plt.savefig(f"{time_str}_waveform_A.png", dpi=600)
print("Waveform_A saved")

plt.figure()
plt.imshow(np.abs(record_waveform_B), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Waveform_B_no_dispersion")
plt.savefig(f"{time_str}_waveform_B.png", dpi=600)
print("Waveform_B saved")

# plot the frequency in a heatmap
plt.figure()
plt.imshow(np.abs(record_freq_A), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Frequency_A_no_dispersion")
plt.savefig(f"{time_str}_frequency_A.png", dpi=600)
print("Frequency_A saved")

plt.figure()
plt.imshow(np.abs(record_freq_B), aspect='auto', \
           extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
# plt.colorbar()
plt.title("Frequency_B_no_dispersion")
plt.savefig(f"{time_str}_frequency_B.png", dpi=600)
print("Frequency_B saved")

# plt.show()

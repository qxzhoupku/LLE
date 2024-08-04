import numpy as np
from numba import jit, objmode
import matplotlib.pyplot as plt
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


def result_plot(record_power_A, record_power_B, record_waveform_A, record_waveform_B, zetas, time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end):
    # Plot power
    plt.figure()
    plt.plot(zetas, record_power_A, label=f'Power A, f_A = {f_A}')
    plt.plot(zetas, record_power_B, label=f'Power B, f_B = {f_B}')
    plt.xlim(zeta_ini, zeta_end)
    plt.title(f"Power, J = {J_back_r}")
    plt.xlabel("detuning")
    plt.legend()
    plt.savefig(f"{time_str}_power.png", dpi=600)

    # Plot waveform heatmap
    record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
    record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)

    # plot final waveform, with mode number as horizontal axis
    plt.figure()
    plt.plot(np.abs(record_waveform_A[-1]), label = "A", alpha = 0.5)
    plt.plot(np.abs(record_waveform_B[-1]), label = "B", alpha = 0.5)
    plt.title("Final Waveform")
    plt.xlabel("mode number")
    plt.ylabel("amplitude")
    plt.savefig(f"{time_str}_final_waveform.png", dpi=600)
    print("Final Waveform saved")

    # plot final spectrum, with mode number as horizontal axis
    plt.figure()
    plt.plot(np.abs(record_freq_A[-1]), label = "A", alpha = 0.5)
    plt.plot(np.abs(record_freq_B[-1]), label = "B", alpha = 0.5)
    plt.title("Final Spectrum")
    plt.xlabel("mode number")
    plt.ylabel("amplitude")
    plt.savefig(f"{time_str}_final_spectrum.png", dpi=600)
    print("Final Spectrum saved")

    record_freq_A = record_freq_A.T
    record_freq_B = record_freq_B.T
    record_waveform_A = record_waveform_A.T
    record_waveform_B = record_waveform_B.T

    plt.figure()
    plt.imshow(np.abs(record_waveform_A), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    # plt.colorbar()
    plt.title("Waveform_A")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_waveform_A.png", dpi=600)
    print("Waveform_A saved")

    plt.figure()
    plt.imshow(np.abs(record_waveform_B), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    # plt.colorbar()
    plt.title("Waveform_B")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_waveform_B.png", dpi=600)
    print("Waveform_B saved")

    # plot the frequency in a heatmap
    plt.figure()
    plt.imshow(np.abs(record_freq_A), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    # plt.colorbar()
    plt.title("Frequency_A")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_frequency_A.png", dpi=600)
    print("Frequency_A saved")

    plt.figure()
    plt.imshow(np.abs(record_freq_B), aspect='auto', \
            extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    # plt.colorbar()
    plt.title("Frequency_B")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_frequency_B.png", dpi=600)
    print("Frequency_B saved")

    # plt.show()




# Initialization
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
    line_A, = ax.plot(np.abs(A))
    line_B, = ax.plot(np.abs(B))
    xs_freq = np.arange(-mode_number / 2, mode_number / 2)
    line_A_freq, = ax_freq.plot(xs_freq, np.abs(A_freq))
    line_B_freq, = ax_freq.plot(xs_freq, np.abs(B_freq))
    line_A_phase, = ax_phase_A.plot(np.angle(A))
    line_B_phase, = ax_phase_B.plot(np.angle(B))

print("Start main loop")
if cProfile_test:
    cProfile.run("main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B)", f"{time_str}_profile.prof")
else:
    main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B)
print("End main loop")

plt.ioff()

# store D_int
np.savetxt(f"{time_str}_D_int.txt", D_int)

result_plot(record_power_A, record_power_B, record_waveform_A, record_waveform_B, zetas, time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end)

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import cProfile

from parameters import mode_number, iter_number, plot_interval, record_interval, zeta_ini, zeta_end, zetas, f_A, f_B, J_back_r, delta_t, D_int, time_str, plot_flag, cProfile_test, noise_flag # type: ignore

# Change directory to output folder
current_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(current_path))
output_path = "../output"
if not os.path.exists(output_path):
    os.mkdir(output_path)
os.chdir(output_path)

# Set random seed for reproducibility
seed = 42
rng = np.random.default_rng(seed)

def noise(mode_number, rng):
    white_noise = rng.standard_normal(mode_number) + 1j * rng.standard_normal(mode_number)
    return white_noise

def cal_power(x):
    return np.sum(np.abs(x)**2) / len(x)

def split_step(A_0, zeta, f, D_int, delta_t, B, J_back_r=0, noise_flag=False):
    B_avg_pow = cal_power(B)
    A_1 = np.exp(1j * (np.abs(A_0)**2 + B_avg_pow) * delta_t) * A_0
    A_1_freq = np.fft.fft(A_1)
    A_2_freq = np.exp(-(1 + 1j * zeta + 1j * D_int) * delta_t) * A_1_freq
    A_2 = np.fft.ifft(A_2_freq)
    A_3 = A_2 + f * delta_t
    A_4 = A_3 + 1j * J_back_r * delta_t * B # backscattering term from backwards mode
    if noise_flag:
        A_4 += noise(mode_number, rng) * 0.0001
    return A_4

def figure_plot(A, B, i, zeta, ax, ax_freq, line_A, line_B, line_A_freq, line_B_freq, line_A_phase, line_B_phase):
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

def main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag):
    record_power_A = np.zeros(iter_number)
    record_power_B = np.zeros(iter_number)
    record_waveform_A = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)
    record_waveform_B = np.zeros((iter_number // record_interval, mode_number), dtype=np.complex128)

    if plot_flag:
        plt.ion()
        fig, axs = plt.subplots(4)
        fig.set_size_inches(5, 7)
        ax, ax_freq, ax_phase_A, ax_phase_B = axs[0], axs[1], axs[2], axs[3]
        line_A, = ax.plot(np.abs(A))
        line_B, = ax.plot(np.abs(B))
        xs_freq = np.arange(-mode_number / 2, mode_number / 2)
        line_A_freq, = ax_freq.plot(xs_freq, np.abs(A_freq))
        line_B_freq, = ax_freq.plot(xs_freq, np.abs(B_freq))
        line_A_phase, = ax_phase_A.plot(np.angle(A))
        line_B_phase, = ax_phase_B.plot(np.angle(B))

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

        if i % plot_interval == 0 and plot_flag:
            figure_plot(A, B, i, zeta, ax, ax_freq, line_A, line_B, line_A_freq, line_B_freq, line_A_phase, line_B_phase)

    plt.ioff()

    # Store data and plot results
    np.savetxt(f"{time_str}_D_int.txt", D_int)
    np.savez(f"{time_str}_results.npz", record_power_A=record_power_A, record_power_B=record_power_B, 
             record_waveform_A=record_waveform_A, record_waveform_B=record_waveform_B)

    plt.figure()
    plt.plot(zetas, record_power_A, label=f'Power A, f_A = {f_A}')
    plt.plot(zetas, record_power_B, label=f'Power B, f_B = {f_B}')
    plt.xlim(zeta_ini, zeta_end)
    plt.title(f"Power, J = {J_back_r}")
    plt.xlabel("detuning")
    plt.legend()
    plt.savefig(f"{time_str}_power.png", dpi=600)

    record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1).T
    record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1).T

    plt.figure()
    plt.imshow(np.abs(record_waveform_A.T), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    plt.title("Waveform_A")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_waveform_A.png", dpi=600)
    print("Waveform_A saved")

    plt.figure()
    plt.imshow(np.abs(record_waveform_B.T), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    plt.title("Waveform_B")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_waveform_B.png", dpi=600)
    print("Waveform_B saved")

    plt.figure()
    plt.imshow(np.abs(record_freq_A), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    plt.title("Frequency_A")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_frequency_A.png", dpi=600)
    print("Frequency_A saved")

    plt.figure()
    plt.imshow(np.abs(record_freq_B), aspect='auto', extent=[zeta_ini, zeta_end, -mode_number / 2, mode_number / 2])
    plt.title("Frequency_B")
    plt.xlabel("detuning")
    plt.savefig(f"{time_str}_frequency_B.png", dpi=600)
    print("Frequency_B saved")

print("Start main loop")
if cProfile_test:
    cProfile.run("main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag)", f"{time_str}_profile.prof")
else:
    main_loop(iter_number, plot_interval, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag)
print("End main loop")

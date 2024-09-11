import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def result_view(record_power_A, record_power_B, record_waveform_A, record_waveform_B, zetas, time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, d_2):
    # Plot power
    plt.figure()
    plt.plot(zetas, record_power_A, label=f'f_A = {f_A}', alpha = 0.7)
    plt.plot(zetas, record_power_B, label=f'f_B = {f_B}', alpha = 0.7)
    plt.xlim(zeta_ini, zeta_end)
    plt.title(f"Power, J = {J_back_r}, d_2 = {d_2}")
    plt.xlabel("detuning")
    plt.legend(loc = "lower left")

    # Plot waveform heatmap
    record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
    record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)

    record_freq_A = record_freq_A.T
    record_freq_B = record_freq_B.T
    record_waveform_A = record_waveform_A.T
    record_waveform_B = record_waveform_B.T

    plt.figure()
    plt.imshow(np.abs(record_waveform_A), aspect='auto', extent=[zeta_ini, zeta_end, mode_number, 0])
    # plt.colorbar()
    plt.title("Waveform_A")
    plt.xlabel("detuning")

    plt.figure()
    plt.imshow(np.abs(record_waveform_B), aspect='auto', extent=[zeta_ini, zeta_end, mode_number, 0])
    # plt.colorbar()
    plt.title("Waveform_B")
    plt.xlabel("detuning")

    # plot the frequency in a heatmap
    plt.figure()
    plt.imshow(np.abs(record_freq_A), aspect='auto', extent=[zeta_ini, zeta_end, mode_number / 2, -mode_number / 2])
    # plt.colorbar()
    plt.title("Frequency_A")
    plt.xlabel("detuning")

    plt.figure()
    plt.imshow(np.abs(record_freq_B), aspect='auto', extent=[zeta_ini, zeta_end, mode_number / 2, -mode_number / 2])
    # plt.colorbar()
    plt.title("Frequency_B")
    plt.xlabel("detuning")

    plt.show()


time_str = sys.argv[1]

os.chdir(os.path.dirname(__file__))
os.chdir("../cache/saved")

record_power_A = np.load(f"{time_str}_record_power_A.npy")
record_power_B = np.load(f"{time_str}_record_power_B.npy")
record_waveform_A = np.load(f"{time_str}_record_waveform_A.npy")
record_waveform_B = np.load(f"{time_str}_record_waveform_B.npy")
# time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, iter_number, power_interval = np.load(f"{time_str}_parameters.npy", allow_pickle=True)
data_dict = np.load(f"{time_str}_parameters_dict.npy", allow_pickle=True).item()
f_A = data_dict["f_A"]
f_B = data_dict["f_B"]
J_back_r = data_dict["J_back_r"]
mode_number = data_dict["mode_number"]
zeta_ini = data_dict["zeta_ini"]
zeta_end = data_dict["zeta_end"]
d_2 = data_dict["d_2"]
zetas = np.load(f"{time_str}_zetas.npy")
print("length of zetas:", len(zetas))
print("length of record_waveform:", len(record_waveform_A))


result_view(record_power_A, record_power_B, record_waveform_A, record_waveform_B, zetas, time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, d_2)


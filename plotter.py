import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def result_plot(record_power_A, record_power_B, record_waveform_A, record_waveform_B, zetas, time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end):
    # Plot power
    plt.figure()
    plt.plot(zetas, record_power_A, label=f'Power A, f_A = {f_A}', alpha = 0.7)
    plt.plot(zetas, record_power_B, label=f'Power B, f_B = {f_B}', alpha = 0.7)
    plt.xlim(zeta_ini, zeta_end)
    plt.title(f"Power, J = {J_back_r}")
    plt.xlabel("detuning")
    plt.legend(loc = "lower left")
    plt.savefig(f"{time_str}_power.png", dpi=600)
    print("Power saved")

    # Plot waveform heatmap
    record_freq_A = np.fft.fftshift(np.fft.fft(record_waveform_A, axis=1), axes=1)
    record_freq_B = np.fft.fftshift(np.fft.fft(record_waveform_B, axis=1), axes=1)

    # plot final waveform, with mode number as horizontal axis
    plt.figure()
    plt.plot(np.abs(record_waveform_A[-1]), label = "A", alpha = 0.7)
    plt.plot(np.abs(record_waveform_B[-1]), label = "B", alpha = 0.7)
    plt.title("Final Waveform")
    plt.xlabel("mode number")
    plt.ylabel("amplitude")
    plt.savefig(f"{time_str}_final_waveform.png", dpi=600)
    print("Final Waveform saved")

    # plot final spectrum, with mode number as horizontal axis
    plt.figure()
    plt.plot(np.abs(record_freq_A[-1]), label = "A", alpha = 0.7)
    plt.plot(np.abs(record_freq_B[-1]), label = "B", alpha = 0.7)
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


time_str = sys.argv[1]

os.chdir(os.path.dirname(__file__))
os.chdir("../cache")

record_power_A = np.load(f"{time_str}_record_power_A.npy")
record_power_B = np.load(f"{time_str}_record_power_B.npy")
record_waveform_A = np.load(f"{time_str}_record_waveform_A.npy")
record_waveform_B = np.load(f"{time_str}_record_waveform_B.npy")
time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end, iter_number, power_interval = np.load(f"{time_str}_parameters.npy", allow_pickle=True)
zetas = np.load(f"{time_str}_zetas.npy")
print("length of zetas:", len(zetas))
print("length of record_waveform:", len(record_waveform_A))

os.chdir("../output")

result_plot(record_power_A, record_power_B, record_waveform_A, record_waveform_B, zetas, time_str, f_A, f_B, J_back_r, mode_number, zeta_ini, zeta_end)


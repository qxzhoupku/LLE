import numpy as np
from numba import jit, objmode
import os
from scipy.ndimage import gaussian_filter1d
import time


mode_number = 2**8
iter_number = 10**5
plot_interval = 5000
record_interval = iter_number // 10000
# zeta is changing every single iteration
zeta_ini = +5.0 - 0.0001
zeta_end = +10.0 + 0.0001
zetas = np.linspace(zeta_ini, zeta_end, iter_number)

f_A = 3
f_B = 0
delta_t = 1e-4 # commonly used time step
delta_t = 1e-5
J_back_r = 2.85


D_int = np.zeros(mode_number, dtype=np.complex128)
for i in range(mode_number):
    D_int[i] = (i - mode_number / 2) ** 2 / 2

D_int = np.fft.ifftshift(D_int)

time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

random_seed = np.random.randint(0, 2**32)
rng = np.random.default_rng(random_seed)


noise_flag = False
# noise_flag = True


os.chdir(os.path.dirname(__file__))
os.chdir("../output")
# save all the variables to a file
with open(f"{time_str}.txt", 'w') as file:
    for var in dir():
        if var in {"np", "file", "time", "D_int"}:
            continue
        if not var.startswith("__") and not var.startswith("_"):
            # print(var)
            file.write(f"{var} = {eval(var)}\n")

print(time_str)


@jit(nopython=True)
def noise(mode_number, rng):
    white_noise = rng.standard_normal(mode_number) + 1j * rng.standard_normal(mode_number)
    return white_noise


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
        A_4 += noise(mode_number, rng) * delta_t
    return A_4


################
# Main loop
@jit(nopython=True)
def main_loop(iter_number, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B):
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
main_loop(iter_number, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng, record_power_A, record_power_B, record_waveform_A, record_waveform_B)
print("End main loop")

# store D_int
np.savetxt(f"{time_str}_D_int.txt", D_int)

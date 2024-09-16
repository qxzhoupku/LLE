import numpy as np
import os
import time


f_A = 5
f_B = 0
d_2 = 0.1
J_back_r = 10.0
noise_level = 1e-6

zeta_ini = -5 - 0.0001
zeta_end = -4 + 0.0001
iter_number = 10**7
mode_number = 2**8
delta_t = 1e-5
random_seed = np.random.randint(0, 2**32)

plot_flag = False
cProfile_test = False
noise_flag = True


plot_interval = 5000
record_interval = iter_number // 10000
power_interval = max(iter_number // 1000000, 1)
zeta_step = (zeta_end - zeta_ini) / (iter_number - 1)
seed_number = -1
D_int = np.zeros(mode_number, dtype=np.complex128)
for i in range(mode_number):
    D_int[i] = (i - mode_number / 2) ** 2 * d_2
D_int = np.fft.ifftshift(D_int)
time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
rng = np.random.default_rng(random_seed)


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

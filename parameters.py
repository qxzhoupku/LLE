import numpy as np
import os
import time

mode_number = 2**8
iter_number = 10**8
plot_interval = 5000
record_interval = iter_number // 10000
power_interval = max(iter_number // 1000000, 1)
# zeta is changing every single iteration
zeta_ini = +6.50 - 0.0001
zeta_end = +6.50 + 0.0001
zeta_step = (zeta_end - zeta_ini) / (iter_number - 1)

f_A = 3
f_B = 0
delta_t = 1e-4 # commonly used time step
delta_t = 1e-5
J_back_r = 2.85

noise_level = 1e-4

seed_number = -1
seed_number = 6


D_int = np.zeros(mode_number, dtype=np.complex128)
for i in range(mode_number):
    D_int[i] = (i - mode_number / 2) ** 2 / 2

D_int = np.fft.ifftshift(D_int)

time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

random_seed = np.random.randint(0, 2**32)
# random_seed = 4125589608
rng = np.random.default_rng(random_seed)


plot_flag = False
# plot_flag = True
cProfile_test = False
# cProfile_test = True
noise_flag = False
noise_flag = True



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

import numpy as np
import os
import time

mode_number = 2**10
iter_number = 10**6
plot_interval = 5000
record_interval = iter_number // 10000
# zeta is changing every single iteration
zeta_ini = -0.5 - 0.0001
zeta_end = +0.5 + 0.0001
zetas = np.linspace(zeta_ini, zeta_end, iter_number)

f_A = 3
f_B = 0
J_back_r = 2.85
# loss_back = 1
# r_back = loss_back * J_back_r**0.5; t_back = loss_back * (1 - J_back_r)**0.5
delta_t = 0.0001 # commonly used time step
# delta_t = 0.01

D_int = np.zeros(mode_number, dtype=np.complex128)
for i in range(mode_number):
    D_int[i] = (i - mode_number / 2) ** 2 / 2

D_int = np.fft.ifftshift(D_int)

time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

random_seed = np.random.randint(0, 2**32)
np.random.seed(random_seed)


plot_flag = False
# plot_flag = True
cProfile_test = False
# cProfile_test = True
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

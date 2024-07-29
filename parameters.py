import numpy as np
import os
import time

mode_number = 2**10
iter_number = 10**7
plot_interval = 5000
record_interval = iter_number // 10000
# zeta is changing every single iteration
zeta_ini = -50
zeta_end = 50
zetas = np.linspace(zeta_ini, zeta_end, iter_number)

f_A = 3
f_B = 0
J_back_r = 2.85
loss_back = 1
r_back = loss_back * J_back_r**0.5; t_back = loss_back * (1 - J_back_r)**0.5
delta_t = 0.0001

D_int = np.zeros(mode_number, dtype=np.complex128)
for i in range(mode_number):
    D_int[i] = (i - mode_number / 2) ** 2 / 2

time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


os.chdir(os.path.dirname(__file__))
os.chdir("../output")
# save all the variables to a file
with open(f"{time_str}.txt", 'w') as file:
    for var in dir():
        if var in {"np", "file", "time"}:
            continue
        if not var.startswith("__") and not var.startswith("_"):
            file.write(f"{var} = {eval(var)}\n")

plot_flag = False
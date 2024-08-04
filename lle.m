clear;
clc;

mode_number = 2^8;
iter_number = 10^6;
plot_interval = 5000;
record_interval = floor(iter_number / 10000);
zeta_ini = 5.0 - 0.0001;
zeta_end = 10.0 + 0.0001;
zetas = linspace(zeta_ini, zeta_end, iter_number);

f_A = 3;
f_B = 0;
% delta_t = 1e-4;
delta_t = 1e-5;
J_back_r = 2.85;

D_int = zeros(1, mode_number) + 1i*zeros(1, mode_number);
for i = 1:mode_number
    D_int(i) = (i - mode_number / 2)^2 / 2;
end

D_int = ifftshift(D_int);

time_str = datetime("now");

rng('shuffle');
random_seed = randi([0, 2^32]);
rng(random_seed);

noise_flag = false;

% Create the output directory if it does not exist
output_dir = fullfile(pwd, '../output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

disp(time_str);

% Define functions
function white_noise = noise(mode_number)
    white_noise = randn(1, mode_number) + 1i * randn(1, mode_number);
end

function power = cal_power(x)
    mode_number = length(x);
    power = sum(abs(x).^2) / mode_number;
end

function A_4 = split_step(A_0, zeta, f, D_int, delta_t, B, B_avg_pow, J_back_r, noise_flag)
    A_1 = exp(1i * (abs(A_0).^2 + B_avg_pow) * delta_t) .* A_0;
    A_1_freq = fft(A_1);
    A_2_freq = exp(-(1 + 1i * zeta + 1i * D_int) * delta_t) .* A_1_freq;
    A_2 = ifft(A_2_freq);
    A_3 = A_2 + f * delta_t;
    A_4 = A_3 + 1i * J_back_r * delta_t * B;
    if noise_flag
        A_4 = A_4 + noise(mode_number) * 0.0001;
    end
end



% Initialization
A = noise(mode_number) * 0.0001;
B = noise(mode_number) * 0.0001;

record_power_A = zeros(1, iter_number);
record_power_B = zeros(1, iter_number);
record_waveform_A = zeros(iter_number / record_interval, mode_number) + 1i * zeros(iter_number / record_interval, mode_number);
record_waveform_B = zeros(iter_number / record_interval, mode_number) + 1i * zeros(iter_number / record_interval, mode_number);

disp('Start main loop');
% Main loop
for i = 1:iter_number
    zeta = zetas(i);
    power_A = cal_power(A);
    power_B = cal_power(B);
    record_power_A(i) = power_A;
    record_power_B(i) = power_B;
    A_new = split_step(A, zeta, f_A, D_int, delta_t, B, power_B, J_back_r, noise_flag);
    B_new = split_step(B, zeta, f_B, D_int, delta_t, A, power_A, J_back_r, noise_flag);
    A = A_new;
    B = B_new;

    if mod(i, record_interval) == 0
        record_waveform_A(floor(i / record_interval), :) = A;
        record_waveform_B(floor(i / record_interval), :) = B;
    end
end
disp('End main loop');



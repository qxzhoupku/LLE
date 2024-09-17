clear;
clc;

f_A = 5;
f_B = 0;
d_2 = 0.04;
J_back_r = 0.0;
noise_level = 1e-6;

zeta_ini = -5.0 - 0.0001;
zeta_end = +40.0 + 0.0001;
iter_number = 10^6;
mode_number = 2^8;
delta_t = 1e-4;
noise_flag = true;
rng('shuffle');
random_seed = randi([0, 2^31]);
rng(random_seed);


record_interval = floor(iter_number / 10000);
zetas = linspace(zeta_ini, zeta_end, iter_number);
D_int = zeros(1, mode_number) + 1i*zeros(1, mode_number);
for i = 1:mode_number
    D_int(i) = (i - mode_number / 2)^2 * d_2;
end
D_int = ifftshift(D_int);
time_str = datetime("now");


disp(time_str);


% Define functions
function white_noise = noise(mode_number, noise_level)
    white_noise = (randn(1, mode_number) + 1i * randn(1, mode_number)) * noise_level;
end

function power = cal_power(x)
    mode_number = length(x);
    power = sum(abs(x).^2) / mode_number;
end

function A_4 = split_step(A_0, zeta, f, D_int, delta_t, B, B_avg_pow, J_back_r, noise_flag, mode_number, noise_level)
    A_1 = exp(1i * (abs(A_0).^2 + 2 * B_avg_pow) * delta_t) .* A_0;
    A_1_freq = fft(A_1);
    A_2_freq = exp(-(1 + 1i * zeta + 1i * D_int) * delta_t) .* A_1_freq;
    A_2 = ifft(A_2_freq);
    A_3 = A_2 + f * delta_t;
    A_4 = A_3 + 1i * J_back_r * delta_t * B;
    if noise_flag
        A_4 = A_4 + noise(mode_number, noise_level) * delta_t;
    end
end


% Initialization
A = noise(mode_number, noise_level);
B = noise(mode_number, noise_level);

record_power_A = zeros(1, iter_number);
record_power_B = zeros(1, iter_number);
record_waveform_A = zeros(floor(iter_number / record_interval), mode_number) + 1i * zeros(floor(iter_number / record_interval), mode_number);
record_waveform_B = zeros(floor(iter_number / record_interval), mode_number) + 1i * zeros(floor(iter_number / record_interval), mode_number);


disp('Start main loop');
% Main loop
time_str = datetime("now");
for i = 1:iter_number
    zeta = zetas(i);
    power_A = cal_power(A);
    power_B = cal_power(B);
    record_power_A(i) = power_A;
    record_power_B(i) = power_B;
    A_new = split_step(A, zeta, f_A, D_int, delta_t, B, power_B, J_back_r, noise_flag, mode_number, noise_level);
    B_new = split_step(B, zeta, f_B, D_int, delta_t, A, power_A, J_back_r, noise_flag, mode_number, noise_level);
    A = A_new;
    B = B_new;

    if mod(i, record_interval) == 0
        record_waveform_A(floor(i / record_interval), :) = A;
        record_waveform_B(floor(i / record_interval), :) = B;
    end
end
% time
time_cost = datetime("now") - time_str;
disp(time_cost);
disp('End main loop');

figure;
plot(record_power_A);
hold on;
plot(record_power_B);
hold off;
title('Power');
xlabel('Iteration');
ylabel('Power');
legend('A', 'B');

figure;
h = heatmap(abs(record_waveform_A), 'Colormap', parula, 'ColorbarVisible', 'on');
title('Logarithmic Waveform A Evolution (Heatmap)');
xlabel('Mode Number');
ylabel('Time Step');
h.XDisplayLabels = repmat("", 1, mode_number);
h.YDisplayLabels = repmat("", size(record_waveform_A, 1), 1);

figure;
h = heatmap(abs(record_waveform_B), 'Colormap', parula, 'ColorbarVisible', 'on');
title('Logarithmic Waveform A Evolution (Heatmap)');
xlabel('Mode Number');
ylabel('Time Step');
h.XDisplayLabels = repmat("", 1, mode_number);
h.YDisplayLabels = repmat("", size(record_waveform_B, 1), 1);


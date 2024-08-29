using FFTW
using Random
using Printf
using Dates

# 参数初始化
const mode_number = 2^8
const iter_number = 10^6
const plot_interval = 5000
const record_interval = iter_number ÷ 10000
const zeta_ini = 5.0 - 0.0001
const zeta_end = 10.0 + 0.0001
const zetas = range(zeta_ini, stop=zeta_end, length=iter_number)

const f_A = 3
const f_B = 0
const delta_t = 1e-5
const J_back_r = 2.85

# 初始化D_int
D_int = Complex{Float64}[((i - mode_number / 2)^2 / 2) for i in 1:mode_number]
D_int = ifftshift(D_int)

# 时间戳
time_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# 随机种子
random_seed = rand(UInt32)
rng = MersenneTwister(random_seed)

# 噪声标志
const noise_flag = false

println("Timestamp: ", time_str)

# 噪声生成函数
function noise(mode_number, rng)
    white_noise = randn(rng, mode_number) .+ 1im * randn(rng, mode_number)
    return white_noise
end

# 计算功率函数
function cal_power(x)
    mode_number = length(x)
    return sum(abs2.(x)) / mode_number
end

# 分步算法
function split_step(A_0, zeta, f, D_int, delta_t, B, B_avg_pow, J_back_r=0, noise_flag=false, rng=GlobalRNG)
    # 计算逐元素的复指数
    phase = exp.(1im * (abs2.(A_0) .+ B_avg_pow) * delta_t)
    A_1 = phase .* A_0
    A_1_freq = fft(A_1)
    exp_factor = exp.(-(1 .+ 1im * zeta .+ 1im * D_int) * delta_t)
    A_2_freq = exp_factor .* A_1_freq
    A_2 = ifft(A_2_freq)
    A_3 = A_2 .+ f * delta_t
    A_4 = A_3 .+ 1im * J_back_r * delta_t * B # backscattering term from backwards mode
    if noise_flag
        A_4 .+= noise(mode_number, rng) * delta_t
    end
    return A_4
end

# 主循环
function main_loop(iter_number, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng)
    record_power_A = zeros(iter_number)
    record_power_B = zeros(iter_number)
    record_waveform_A = [zeros(Complex{Float64}, mode_number) for _ in 1:(iter_number ÷ record_interval)]
    record_waveform_B = [zeros(Complex{Float64}, mode_number) for _ in 1:(iter_number ÷ record_interval)]
    
    for i in 1:iter_number
        zeta = zetas[i]
        power_A = cal_power(A)
        power_B = cal_power(B)
        record_power_A[i] = power_A
        record_power_B[i] = power_B
        A_new = split_step(A, zeta, f_A, D_int, delta_t, B, power_B, J_back_r, noise_flag, rng)
        B_new = split_step(B, zeta, f_B, D_int, delta_t, A, power_A, J_back_r, noise_flag, rng)
        A, B = A_new, B_new

        if i % record_interval == 0
            record_waveform_A[i ÷ record_interval] = A
            record_waveform_B[i ÷ record_interval] = B
        end
    end
    
    return record_power_A, record_power_B, record_waveform_A, record_waveform_B
end

# 初始化
A = noise(mode_number, rng) * 0.0001
B = noise(mode_number, rng) * 0.0001
A_freq = fftshift(fft(A))
B_freq = fftshift(fft(B))

println("Start main loop")
record_power_A, record_power_B, record_waveform_A, record_waveform_B = main_loop(iter_number, record_interval, zetas, A, B, f_A, f_B, D_int, delta_t, J_back_r, noise_flag, rng)
println("End main loop")

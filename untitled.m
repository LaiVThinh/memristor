%% ==================== MAIN SCRIPT ====================
clear all; close all; clc;

%% ==================== KIỂM TRA VÀ KHỞI ĐỘNG GPU ====================
fprintf('=== KIỂM TRA GPU ===\n');
try
    g = gpuDevice;
    fprintf('GPU được phát hiện: %s\n', g.Name);
catch
    error('Không tìm thấy GPU. Vui lòng đảm bảo bạn đã cài đặt Parallel Computing Toolbox.');
end

%% ==================== LOAD DATA AND MODELS (CPU) ====================
fprintf('\n=== LOADING DATA AND MODELS ===\n');

% Load dữ liệu MNIST
load('mnist_test_data.mat');
load('sgd_int4.mat');  % Baseline model

% Load RWP model
rwp_data = load('sam_int4.mat');
W1_rwp = rwp_data.W1;
W2_rwp = rwp_data.W2;
b1_rwp = rwp_data.b1;
b2_rwp = rwp_data.b2;

% Chuẩn bị dữ liệu
Ytest = Ytest + 1;  % Chuyển labels từ 0-9 sang 1-10
Xtest = Xtest';

% Điều chỉnh kích thước bias
if size(b1, 1) == 1, b1 = b1'; end
if size(b2, 1) == 1, b2 = b2'; end
if size(b1_rwp, 1) == 1, b1_rwp = b1_rwp'; end
if size(b2_rwp, 1) == 1, b2_rwp = b2_rwp'; end

% Điều chỉnh kích thước weights
if size(W1, 1) == 784 && size(W1, 2) == 512, W1 = W1'; end
if size(W2, 1) == 512 && size(W2, 2) == 10, W2 = W2'; end
if size(W1_rwp, 1) == 784 && size(W1_rwp, 2) == 512, W1_rwp = W1_rwp'; end
if size(W2_rwp, 1) == 512 && size(W2_rwp, 2) == 10, W2_rwp = W2_rwp'; end

num_test = 1000;
fprintf('Số lượng mẫu test: %d\n', num_test);

%% ==================== CHUYỂN DỮ LIỆU LÊN GPU ====================
fprintf('\n=== ĐƯA DỮ LIỆU VÀ MÔ HÌNH LÊN GPU ===\n');

Xtest_gpu = gpuArray(single(Xtest));
Ytest_gpu = gpuArray(int32(Ytest(:)));

W1_gpu = gpuArray(single(W1));
W2_gpu = gpuArray(single(W2));
b1_gpu = gpuArray(single(b1));
b2_gpu = gpuArray(single(b2));

W1_rwp_gpu = gpuArray(single(W1_rwp));
W2_rwp_gpu = gpuArray(single(W2_rwp));
b1_rwp_gpu = gpuArray(single(b1_rwp));
b2_rwp_gpu = gpuArray(single(b2_rwp));

fprintf('Dữ liệu và mô hình đã được chuyển lên GPU.\n');

%% ==================== TEST ORIGINAL MODELS (GPU - IDEAL ReLU) ====================
fprintf('\n=== TESTING ORIGINAL MODELS (GPU - IDEAL ReLU) ===\n');

X_batch_test = Xtest_gpu(1:num_test, :);
Y_batch_test = Ytest_gpu(1:num_test);

% Baseline model với ReLU lý tưởng
z1 = X_batch_test * W1_gpu' + b1_gpu';
a1_ideal = max(0, z1);  % ReLU lý tưởng
z2 = a1_ideal * W2_gpu' + b2_gpu';
[~, pred_baseline] = max(z2, [], 2);
accuracy_baseline_ideal = gather(sum(pred_baseline == Y_batch_test)) / num_test * 100;
fprintf('Baseline MLP (Ideal ReLU): %.3f%%\n', accuracy_baseline_ideal);

% RWP model với ReLU lý tưởng
z1_rwp = X_batch_test * W1_rwp_gpu' + b1_rwp_gpu';
a1_rwp_ideal = max(0, z1_rwp);  % ReLU lý tưởng
z2_rwp = a1_rwp_ideal * W2_rwp_gpu' + b2_rwp_gpu';
[~, pred_rwp] = max(z2_rwp, [], 2);
accuracy_rwp_ideal = gather(sum(pred_rwp == Y_batch_test)) / num_test * 100;
fprintf('RWP Robust (Ideal ReLU):   %.3f%%\n', accuracy_rwp_ideal);

%% ==================== HARDWARE PARAMETER SETUP ====================
fprintf('\n=== THIẾT LẬP THAM SỐ PHẦN CỨNG ===\n');

G_min = 1e-6;    % Điện dẫn tối thiểu (Siemens)
G_max = 1e-3;    % Điện dẫn tối đa (Siemens)
G_mid = (G_min + G_max) / 2;

% Tìm trọng số lớn nhất để tính R0
max_weight = max([max(abs(W1(:))), max(abs(W2(:)))]);
R0 = max_weight / (G_max - G_mid);  % Điện trở tham chiếu
RB = 1 / G_mid;                     % Điện trở cơ sở
R_F1 = 1e3;                         % Điện trở phản hồi 1
R_F2 = 1e3;                         % Điện trở phản hồi 2

fprintf('Tham số phần cứng:\n');
fprintf('  G_min: %.2e S, G_max: %.2e S\n', G_min, G_max);
fprintf('  R0: %.3f kΩ, RB: %.3f kΩ\n', R0/1000, RB/1000);
fprintf('  R_F1: %.3f kΩ, R_F2: %.3f kΩ\n', R_F1/1000, R_F2/1000);

%% ==================== CONVERT WEIGHTS TO CONDUCTANCE (GPU) ====================
fprintf('\n=== CHUYỂN ĐỔI TRỌNG SỐ THÀNH ĐIỆN DẪN (GPU) ===\n');

% Baseline model
G1_base_gpu = 1/RB - W1_gpu/R0;
G2_base_gpu = 1/RB - W2_gpu/R0;
G1_base_gpu = max(G_min, min(G_max, G1_base_gpu));
G2_base_gpu = max(G_min, min(G_max, G2_base_gpu));

% RWP model
G1_rwp_gpu = 1/RB - W1_rwp_gpu/R0;
G2_rwp_gpu = 1/RB - W2_rwp_gpu/R0;
G1_rwp_gpu = max(G_min, min(G_max, G1_rwp_gpu));
G2_rwp_gpu = max(G_min, min(G_max, G2_rwp_gpu));

fprintf('Phạm vi điện dẫn:\n');
fprintf('  Baseline - G1: [%.2e, %.2e] S\n', min(G1_base_gpu(:)), max(G1_base_gpu(:)));
fprintf('  Baseline - G2: [%.2e, %.2e] S\n', min(G2_base_gpu(:)), max(G2_base_gpu(:)));
fprintf('  RWP      - G1: [%.2e, %.2e] S\n', min(G1_rwp_gpu(:)), max(G1_rwp_gpu(:)));
fprintf('  RWP      - G2: [%.2e, %.2e] S\n', min(G2_rwp_gpu(:)), max(G2_rwp_gpu(:)));

%% ==================== IDEAL HARDWARE FORWARD PASS (WITHOUT MOSFET) ====================
fprintf('\n=== MÔ PHỎNG PHẦN CỨNG LÝ TƯỞNG (KHÔNG MOSFET - IDEAL ReLU) ===\n');

acc_base_hardware_ideal = hardware_forward_pass_ideal_relu(...
    G1_base_gpu, G2_base_gpu, b1_gpu, b2_gpu, ...
    X_batch_test, Y_batch_test, R0, RB, R_F1, R_F2);

acc_rwp_hardware_ideal = hardware_forward_pass_ideal_relu(...
    G1_rwp_gpu, G2_rwp_gpu, b1_rwp_gpu, b2_rwp_gpu, ...
    X_batch_test, Y_batch_test, R0, RB, R_F1, R_F2);

fprintf('Kết quả mô phỏng phần cứng với ReLU lý tưởng:\n');
fprintf('  Baseline: %.3f%% (Phần mềm: %.3f%%)\n', acc_base_hardware_ideal, accuracy_baseline_ideal);
fprintf('  RWP:      %.3f%% (Phần mềm: %.3f%%)\n', acc_rwp_hardware_ideal, accuracy_rwp_ideal);

%% ==================== THÊM NHIỄU GAUSS CHUẨN ====================
fprintf('\n=== THÊM NHIỄU GAUSS CHUẨN (ZERO-MEAN GAUSSIAN NOISE) ===\n');

% Tham số nhiễu Gauss chuẩn
noise_levels = [0.0, 3, 6, 9, 12, 15];  % Độ lệch chuẩn (%)
num_gaussian_test = 1000;

X_gauss_batch = X_batch_test(1:num_gaussian_test, :);
Y_gauss_batch = Y_batch_test(1:num_gaussian_test);

% Test với nhiễu Gauss chuẩn
num_monte_carlo_gauss = 500;

fprintf('\nNhiễu Gauss (%% std) | Baseline | RWP      | Improvement\n');
fprintf('---------------------|----------|----------|------------\n');

baseline_gauss_results = zeros(length(noise_levels), 1);
rwp_gauss_results = zeros(length(noise_levels), 1);

for noise_idx = 1:length(noise_levels)
    noise_std_percent = noise_levels(noise_idx);
    
    % Baseline với nhiễu Gauss chuẩn
    acc_base_total = 0;
    for mc = 1:num_monte_carlo_gauss
        G1_gauss_base = add_gaussian_noise(G1_base_gpu, noise_std_percent, G_min, G_max);
        G2_gauss_base = add_gaussian_noise(G2_base_gpu, noise_std_percent, G_min, G_max);
        
        acc_base = hardware_forward_pass_ideal_relu(...
            G1_gauss_base, G2_gauss_base, b1_gpu, b2_gpu, ...
            X_gauss_batch, Y_gauss_batch, R0, RB, R_F1, R_F2);
        acc_base_total = acc_base_total + acc_base;
    end
    acc_base_avg = acc_base_total / num_monte_carlo_gauss;
    baseline_gauss_results(noise_idx) = acc_base_avg;
    
    % RWP với nhiễu Gauss chuẩn
    acc_rwp_total = 0;
    for mc = 1:num_monte_carlo_gauss
        G1_gauss_rwp = add_gaussian_noise(G1_rwp_gpu, noise_std_percent, G_min, G_max);
        G2_gauss_rwp = add_gaussian_noise(G2_rwp_gpu, noise_std_percent, G_min, G_max);
        
        acc_rwp = hardware_forward_pass_ideal_relu(...
            G1_gauss_rwp, G2_gauss_rwp, b1_rwp_gpu, b2_rwp_gpu, ...
            X_gauss_batch, Y_gauss_batch, R0, RB, R_F1, R_F2);
        acc_rwp_total = acc_rwp_total + acc_rwp;
    end
    acc_rwp_avg = acc_rwp_total / num_monte_carlo_gauss;
    rwp_gauss_results(noise_idx) = acc_rwp_avg;
    
    improvement = acc_rwp_avg - acc_base_avg;
    
    fprintf('        %4.1f%%       |  %6.2f%% | %6.2f%% | %+7.2f%%\n', ...
        noise_std_percent, acc_base_avg, acc_rwp_avg, improvement);
end

%% ==================== THÊM NHIỄU LOG-NORMAL ====================
fprintf('\n=== THÊM NHIỄU LOG-NORMAL (BIẾN THIÊN PHẦN CỨNG) ===\n');

% Tham số nhiễu Log-normal
mu_log = 0;       % Giá trị trung bình của log(G) - thường đặt = 0
sigma_log_levels = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15];  % Các mức độ nhiễu

num_log_normal_test = 1000;
X_log_batch = X_batch_test(1:num_log_normal_test, :);
Y_log_batch = Y_batch_test(1:num_log_normal_test);

% Test với nhiễu Log-normal
num_monte_carlo_log = 50;

fprintf('\nĐộ lệch Log-normal | Baseline | RWP      | Improvement\n');
fprintf('-------------------|----------|----------|------------\n');

baseline_log_results = zeros(length(sigma_log_levels), 1);
rwp_log_results = zeros(length(sigma_log_levels), 1);

for sigma_idx = 1:length(sigma_log_levels)
    sigma_current = sigma_log_levels(sigma_idx);
    
    % Baseline với nhiễu Log-normal
    acc_base_total = 0;
    for mc = 1:num_monte_carlo_log
        G1_log_base = add_lognormal_noise(G1_base_gpu, sigma_current, mu_log, G_min, G_max);
        G2_log_base = add_lognormal_noise(G2_base_gpu, sigma_current, mu_log, G_min, G_max);
        
        acc_base = hardware_forward_pass_ideal_relu(...
            G1_log_base, G2_log_base, b1_gpu, b2_gpu, ...
            X_log_batch, Y_log_batch, R0, RB, R_F1, R_F2);
        acc_base_total = acc_base_total + acc_base;
    end
    acc_base_avg = acc_base_total / num_monte_carlo_log;
    baseline_log_results(sigma_idx) = acc_base_avg;
    
    % RWP với nhiễu Log-normal
    acc_rwp_total = 0;
    for mc = 1:num_monte_carlo_log
        G1_log_rwp = add_lognormal_noise(G1_rwp_gpu, sigma_current, mu_log, G_min, G_max);
        G2_log_rwp = add_lognormal_noise(G2_rwp_gpu, sigma_current, mu_log, G_min, G_max);
        
        acc_rwp = hardware_forward_pass_ideal_relu(...
            G1_log_rwp, G2_log_rwp, b1_rwp_gpu, b2_rwp_gpu, ...
            X_log_batch, Y_log_batch, R0, RB, R_F1, R_F2);
        acc_rwp_total = acc_rwp_total + acc_rwp;
    end
    acc_rwp_avg = acc_rwp_total / num_monte_carlo_log;
    rwp_log_results(sigma_idx) = acc_rwp_avg;
    
    improvement = acc_rwp_avg - acc_base_avg;
    
    fprintf('       σ=%.3f      |  %6.2f%% | %6.2f%% | %+7.2f%%\n', ...
        sigma_current, acc_base_avg, acc_rwp_avg, improvement);
end

%% ==================== THÊM NHIỄU ĐỒNG ĐỀU (UNIFORM NOISE) ====================
fprintf('\n=== THÊM NHIỄU ĐỒNG ĐỀU (UNIFORM NOISE) ===\n');

% Tham số nhiễu Uniform: G_noisy = G * (1 + U(-a, a))
% với a là biên độ nhiễu tính theo phần trăm của G
uniform_levels = [0.0, 3, 6, 9, 12, 15];  % Biên độ nhiễu (%)
num_uniform_test = 1000;

X_uniform_batch = X_batch_test(1:num_uniform_test, :);
Y_uniform_batch = Y_batch_test(1:num_uniform_test);

num_monte_carlo_uniform = 500;

fprintf('\nNhiễu Uniform (%% amp) | Baseline | RWP      | Improvement\n');
fprintf('----------------------|----------|----------|------------\n');

baseline_uniform_results = zeros(length(uniform_levels), 1);
rwp_uniform_results = zeros(length(uniform_levels), 1);

for noise_idx = 1:length(uniform_levels)
    noise_amp_percent = uniform_levels(noise_idx);
    
    % Baseline với nhiễu Uniform
    acc_base_total = 0;
    for mc = 1:num_monte_carlo_uniform
        G1_uniform_base = add_uniform_noise(G1_base_gpu, noise_amp_percent, G_min, G_max);
        G2_uniform_base = add_uniform_noise(G2_base_gpu, noise_amp_percent, G_min, G_max);
        
        acc_base = hardware_forward_pass_ideal_relu(...
            G1_uniform_base, G2_uniform_base, b1_gpu, b2_gpu, ...
            X_uniform_batch, Y_uniform_batch, R0, RB, R_F1, R_F2);
        acc_base_total = acc_base_total + acc_base;
    end
    acc_base_avg = acc_base_total / num_monte_carlo_uniform;
    baseline_uniform_results(noise_idx) = acc_base_avg;
    
    % RWP với nhiễu Uniform
    acc_rwp_total = 0;
    for mc = 1:num_monte_carlo_uniform
        G1_uniform_rwp = add_uniform_noise(G1_rwp_gpu, noise_amp_percent, G_min, G_max);
        G2_uniform_rwp = add_uniform_noise(G2_rwp_gpu, noise_amp_percent, G_min, G_max);
        
        acc_rwp = hardware_forward_pass_ideal_relu(...
            G1_uniform_rwp, G2_uniform_rwp, b1_rwp_gpu, b2_rwp_gpu, ...
            X_uniform_batch, Y_uniform_batch, R0, RB, R_F1, R_F2);
        acc_rwp_total = acc_rwp_total + acc_rwp;
    end
    acc_rwp_avg = acc_rwp_total / num_monte_carlo_uniform;
    rwp_uniform_results(noise_idx) = acc_rwp_avg;
    
    improvement = acc_rwp_avg - acc_base_avg;
    
    fprintf('         %4.1f%%        |  %6.2f%% | %6.2f%% | %+7.2f%%\n', ...
        noise_amp_percent, acc_base_avg, acc_rwp_avg, improvement);
end

%% ==================== WORST-CASE WEIGHT PERTURBATION (ADVERSARIAL WEIGHT NOISE) ====================
fprintf('\n=== WORST-CASE WEIGHT PERTURBATION (ADVERSARIAL WEIGHT NOISE) ===\n');

% Tham số
epsilon_levels = [0.002, 0.004, 0.006, 0.008, 0.01];  % Các mức epsilon
num_adv_test = 500;  % Số lượng mẫu dùng để tính gradient

% Lấy batch dữ liệu trên GPU
X_adv_batch = X_batch_test(1:num_adv_test, :);
Y_adv_batch = Y_batch_test(1:num_adv_test);

% Chuyển về CPU để tính gradient
X_cpu = gather(X_adv_batch);
Y_cpu = gather(Y_adv_batch);

% One-hot encoding
Y_onehot = zeros(num_adv_test, 10);
for i = 1:num_adv_test
    Y_onehot(i, Y_cpu(i)) = 1;
end

% Weights hiện tại (đã gather)
W1_base = gather(W1_gpu);
W2_base = gather(W2_gpu);
b1_base = gather(b1_gpu);
b2_base = gather(b2_gpu);

W1_rwp = gather(W1_rwp_gpu);
W2_rwp = gather(W2_rwp_gpu);
b1_rwp = gather(b1_rwp_gpu);
b2_rwp = gather(b2_rwp_gpu);

fprintf('\nEpsilon | Baseline Clean | Baseline Adv | RWP Clean | RWP Adv | Improvement (RWP - Baseline)\n');
fprintf('--------|---------------|--------------|-----------|---------|-----------------------------\n');

for idx = 1:length(epsilon_levels)
    eps = epsilon_levels(idx);

    % ---- Baseline: tính gradient và tạo weight nhiễu ----
    [grad_W1_base, grad_W2_base] = compute_weight_gradients(...
        X_cpu, Y_onehot, W1_base, W2_base, b1_base, b2_base);
    
    W1_base_adv = W1_base + eps * sign(grad_W1_base);
    W2_base_adv = W2_base + eps * sign(grad_W2_base);
    
    % ---- RWP: tính gradient và tạo weight nhiễu ----
    [grad_W1_rwp, grad_W2_rwp] = compute_weight_gradients(...
        X_cpu, Y_onehot, W1_rwp, W2_rwp, b1_rwp, b2_rwp);
    
    W1_rwp_adv = W1_rwp + eps * sign(grad_W1_rwp);
    W2_rwp_adv = W2_rwp + eps * sign(grad_W2_rwp);
    
    % ---- Chuyển W_adv sang G ----
    G1_base_adv = 1/RB - W1_base_adv/R0;
    G2_base_adv = 1/RB - W2_base_adv/R0;
    G1_base_adv = max(G_min, min(G_max, G1_base_adv));
    G2_base_adv = max(G_min, min(G_max, G2_base_adv));
    
    G1_rwp_adv = 1/RB - W1_rwp_adv/R0;
    G2_rwp_adv = 1/RB - W2_rwp_adv/R0;
    G1_rwp_adv = max(G_min, min(G_max, G1_rwp_adv));
    G2_rwp_adv = max(G_min, min(G_max, G2_rwp_adv));
    
    % Chuyển lên GPU
    G1_base_adv_gpu = gpuArray(single(G1_base_adv));
    G2_base_adv_gpu = gpuArray(single(G2_base_adv));
    G1_rwp_adv_gpu = gpuArray(single(G1_rwp_adv));
    G2_rwp_adv_gpu = gpuArray(single(G2_rwp_adv));
    
    % ---- Đánh giá accuracy trên hardware với weights gốc (clean) và weights nhiễu (adv) ----
    acc_base_clean = hardware_forward_pass_ideal_relu(...
        G1_base_gpu, G2_base_gpu, b1_gpu, b2_gpu, ...
        X_adv_batch, Y_adv_batch, R0, RB, R_F1, R_F2);
    
    acc_base_adv = hardware_forward_pass_ideal_relu(...
        G1_base_adv_gpu, G2_base_adv_gpu, b1_gpu, b2_gpu, ...
        X_adv_batch, Y_adv_batch, R0, RB, R_F1, R_F2);
    
    acc_rwp_clean = hardware_forward_pass_ideal_relu(...
        G1_rwp_gpu, G2_rwp_gpu, b1_rwp_gpu, b2_rwp_gpu, ...
        X_adv_batch, Y_adv_batch, R0, RB, R_F1, R_F2);
    
    acc_rwp_adv = hardware_forward_pass_ideal_relu(...
        G1_rwp_adv_gpu, G2_rwp_adv_gpu, b1_rwp_gpu, b2_rwp_gpu, ...
        X_adv_batch, Y_adv_batch, R0, RB, R_F1, R_F2);
    
    improvement = acc_rwp_adv - acc_base_adv;
    
    fprintf('  %.3f   |     %6.2f%%   |    %6.2f%%   |  %6.2f%%  | %6.2f%% |      %+6.2f%%\n', ...
        eps, acc_base_clean, acc_base_adv, acc_rwp_clean, acc_rwp_adv, improvement);
end


%% ==================== TẤT CẢ CÁC HÀM ĐƯỢC ĐẶT Ở ĐÂY ====================

% HÀM CHÍNH: Hardware forward pass với ReLU lý tưởng
function accuracy = hardware_forward_pass_ideal_relu(G1, G2, b1, b2, X_batch, Y_batch, R0, RB, R_F1, R_F2)
    if ~isa(G1, 'gpuArray')
        G1 = gpuArray(single(G1));
        G2 = gpuArray(single(G2));
        b1 = gpuArray(single(b1));
        b2 = gpuArray(single(b2));
        X_batch = gpuArray(single(X_batch));
        Y_batch = gpuArray(int32(Y_batch));
    end
    
    num_test = size(X_batch, 1);
    
    % Layer 1 - Memristor crossbar
    V_F1_all = -sum(X_batch, 2) * (R_F1 / RB);
    sum_current_mem1_batch = X_batch * G1';
    V_hidden_pre_bias = -[R0 * sum_current_mem1_batch + (R0/R_F2) * V_F1_all];
    V_hidden = V_hidden_pre_bias + b1';
    
    % IDEAL ReLU (không có MOSFET)
    a1 = max(V_hidden, 0);
    
    % Layer 2
    V_F2_all = -sum(a1, 2) * (R_F1 / RB);
    sum_current_mem2_batch = a1 * G2';
    V_output_pre_bias = -[R0 * sum_current_mem2_batch + (R0/R_F2) * V_F2_all];
    V_output = V_output_pre_bias + b2';
    
    [~, pred] = max(V_output, [], 2);
    accuracy = gather(sum(pred == Y_batch)) / num_test * 100;
end

% HÀM THÊM NHIỄU GAUSS CHUẨN (Zero-mean Gaussian Noise)
function G_noisy = add_gaussian_noise(G, noise_std_percent, G_min, G_max)
    % G_noisy = G * (1 + N(0, σ²))
    noise_std = noise_std_percent / 100;
    noise_gaussian = randn(size(G), 'gpuArray') * noise_std;
    G_noisy = G .* (1 + noise_gaussian);
    G_noisy = max(G_min, min(G_max, G_noisy));
end

% HÀM THÊM NHIỄU LOG-NORMAL
function G_noisy = add_lognormal_noise(G, sigma_log, mu_log, G_min, G_max)
    % G_noisy = G * exp(sigma * N(0,1) + mu)
    noise_gaussian = randn(size(G), 'gpuArray');
    log_normal_factor = exp(sigma_log * noise_gaussian + mu_log);
    G_noisy = G .* log_normal_factor;
    G_noisy = max(G_min, min(G_max, G_noisy));
end

% HÀM THÊM NHIỄU ĐỒNG ĐỀU (Uniform Noise)
function G_noisy = add_uniform_noise(G, noise_amp_percent, G_min, G_max)
    % G_noisy = G * (1 + U(-a, a))
    % với a = noise_amp_percent / 100
    % U(-a, a) = 2*a*(rand - 0.5) = a*(2*rand - 1)
    noise_amp = noise_amp_percent / 100;
    noise_uniform = noise_amp * (2 * rand(size(G), 'gpuArray') - 1);
    G_noisy = G .* (1 + noise_uniform);
    G_noisy = max(G_min, min(G_max, G_noisy));
end

%% HÀM TÍNH GRADIENT THEO TRỌNG SỐ
function [grad_W1, grad_W2] = compute_weight_gradients(X, Y_onehot, W1, W2, b1, b2)
    % X: num_samples x 784
    % Y_onehot: num_samples x 10
    % W1: 512 x 784
    % W2: 10 x 512
    % b1: 512 x 1
    % b2: 10 x 1
    
    num_samples = size(X, 1);
    
    % Forward pass
    z1 = X * W1';  % num_samples x 512
    a1 = max(0, z1);  % ReLU
    z2 = a1 * W2';  % num_samples x 10
    
    % Softmax
    exp_z2 = exp(z2 - max(z2, [], 2));
    softmax_probs = exp_z2 ./ sum(exp_z2, 2);
    
    % Gradient tại đầu ra (cross-entropy)
    dL_dz2 = softmax_probs - Y_onehot;  % num_samples x 10
    
    % Gradient cho W2 (10x512)
    grad_W2 = (dL_dz2' * a1) / num_samples;  % trung bình theo batch
    
    % Gradient cho lớp ẩn
    dL_da1 = dL_dz2 * W2;  % num_samples x 512
    dL_dz1 = dL_da1;
    dL_dz1(z1 <= 0) = 0;  % ReLU derivative
    
    % Gradient cho W1 (512x784)
    grad_W1 = (dL_dz1' * X) / num_samples;
end

% HÀM LƯỢNG TỬ HÓA CONDUCTANCE (HLS)
function G_quant = quantize_conductance(G, G_min, G_max, num_levels)
    levels = linspace(G_min, G_max, num_levels);
    G_cpu = gather(G);
    G_quant_cpu = zeros(size(G_cpu));
    
    for i = 1:numel(G_cpu)
        [~, idx] = min(abs(G_cpu(i) - levels));
        G_quant_cpu(i) = levels(idx);
    end
    
    G_quant = gpuArray(single(G_quant_cpu));
end
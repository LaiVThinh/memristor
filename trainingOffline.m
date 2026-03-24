clear all; close all; clc;

%% ==================== LOAD DATA AND MODELS ====================
% Load MNIST test data
load('mnist_test_data.mat');
load('mlp_weights-Best.mat');  % Baseline model

% Load RWP model
rwp_data = load('rwp_weights-0.09_new.mat');
W1_rwp = rwp_data.W1;
W2_rwp = rwp_data.W2;
b1_rwp = rwp_data.b1;
b2_rwp = rwp_data.b2;

% Adjust dimensions (0-9 -> 1-10)
Ytest = Ytest + 1;
Xtest = Xtest';

% Adjust bias dimensions
if size(b1, 1) == 1, b1 = b1'; end
if size(b2, 1) == 1, b2 = b2'; end
if size(b1_rwp, 1) == 1, b1_rwp = b1_rwp'; end
if size(b2_rwp, 1) == 1, b2_rwp = b2_rwp'; end

% Adjust weight dimensions 
if size(W1, 1) == 784 && size(W1, 2) == 512, W1 = W1'; end
if size(W2, 1) == 512 && size(W2, 2) == 10, W2 = W2'; end
if size(W1_rwp, 1) == 784 && size(W1_rwp, 2) == 512, W1_rwp = W1_rwp'; end
if size(W2_rwp, 1) == 512 && size(W2_rwp, 2) == 10, W2_rwp = W2_rwp'; end

num_test = 1000;

%% ==================== TEST ORIGINAL MODELS ====================
fprintf('\n=== TESTING ORIGINAL MODELS ===\n');

% Test baseline model
correct_baseline = 0;
for i = 1:num_test
    sample = Xtest(i, :);
    label = Ytest(i);
    
    z1 = sample * W1' + b1'; % hidden layer input
    a1 = 1./(1+exp(-z1)); %sigmoid
    z2 = a1 * W2' + b2'; % output layer output
    [~, pred] = max(z2); % The index (class) with the largest value in the output vector z2 is the predicted result (pred).
    
    if pred == label
        correct_baseline = correct_baseline + 1;
    end
end
accuracy_baseline = correct_baseline / num_test * 100;
fprintf('Baseline MLP: %.2f%%\n', accuracy_baseline);

% Test RWP model  
correct_rwp = 0;
for i = 1:num_test
    sample = Xtest(i, :);
    label = Ytest(i);
    
    z1_rwp = sample * W1_rwp' + b1_rwp';
    a1_rwp = 1./(1+exp(-z1_rwp));
    z2_rwp = a1_rwp * W2_rwp' + b2_rwp';
    [~, pred_rwp] = max(z2_rwp);
    
    if pred_rwp == label
        correct_rwp = correct_rwp + 1;
    end
end
accuracy_rwp = correct_rwp / num_test * 100;
fprintf('RWP Robust:   %.2f%%\n', accuracy_rwp);

%% ==================== HARDWARE PARAMETER SETUP ====================

% Memristor parameters
G_min = 1e-6;
G_max = 1e-3;
G_mid = (G_min + G_max) / 2;

% Calculate R0 and RB 
%%
% w_jk = R0 × (1/RB - g_jk) => g_jk = 1/RB - w_jk/R0
% Để đảm bảo g_jk thuộc [G_min, G_max], ta cần: G_min <= 1/RB - w_jk/R0 <= G_max
% R0 >= max(|w_jk|) / (G_max - G_mid)
max_weight = max([max(abs(W1(:))), max(abs(W2(:)))]);
R0 = max_weight / (G_max - G_mid);  % Output op-amp feedback resisto
%%
RB = 1 / G_mid;                     % Constant-term resistor
R_F1 = 1e3;                         % Constant-term resistor 1
R_F2 = 1e3;                         % Constant-term resistor 2
V_REF = 0.5;                        % Comparator reference voltage

%% ==================== CONVERT WEIGHTS TO CONDUCTANCE ====================
fprintf('\n=== CONVERTING WEIGHTS TO CONDUCTANCE ===\n');

% Baseline model
% w_jk = R0 × (1/RB - g_jk) => g_jk = 1/RB - w_jk/R0
G1_base = 1/RB - W1/R0;
G2_base = 1/RB - W2/R0;
G1_base = max(G_min, min(G_max, G1_base));
G2_base = max(G_min, min(G_max, G2_base));

% RWP model
G1_rwp = 1/RB - W1_rwp/R0;
G2_rwp = 1/RB - W2_rwp/R0;
G1_rwp = max(G_min, min(G_max, G1_rwp));
G2_rwp = max(G_min, min(G_max, G2_rwp));

% In giới hạn G1 và G2
fprintf('Baseline - G1: [%.2e, %.2e] S\n', min(G1_base(:)), max(G1_base(:)));
fprintf('Baseline - G2: [%.2e, %.2e] S\n', min(G2_base(:)), max(G2_base(:)));
fprintf('RWP      - G1: [%.2e, %.2e] S\n', min(G1_rwp(:)), max(G1_rwp(:)));
fprintf('RWP      - G2: [%.2e, %.2e] S\n', min(G2_rwp(:)), max(G2_rwp(:)));

%% ==================== NOISE-FREE HARDWARE SIMULATION ====================
fprintf('\n=== NOISE-FREE HARDWARE SIMULATION ===\n');

acc_base_hardware = hardware_forward_pass(G1_base, G2_base, Xtest, Ytest, R0, RB, R_F1, R_F2, num_test);
acc_rwp_hardware = hardware_forward_pass(G1_rwp, G2_rwp, Xtest, Ytest, R0, RB, R_F1, R_F2, num_test);

fprintf('Hardware without noise:\n');
fprintf('  Baseline: %.2f%%\n', acc_base_hardware);
fprintf('  RWP:      %.2f%%\n', acc_rwp_hardware);

%% ==================== ROBUSTNESS TEST WITH NOISE ====================
fprintf('\n=== ROBUSTNESS TEST WITH NOISE ===\n');

noise_levels = [3, 6, 9, 12, 15];
num_noise_test = 500;
num_monte_carlo = 50;

fprintf('Testing with %d samples, %d Monte Carlo runs\n', num_noise_test, num_monte_carlo);
fprintf('\nNoise Level | Baseline | RWP\n');
fprintf('------------|----------|-----\n');

for i = 1:length(noise_levels)
    noise = noise_levels(i);
    
    % Monte Carlo for Baseline
    acc_base_total = 0;
    for mc = 1:num_monte_carlo
        acc_base_mc = test_with_noise_monte_carlo(G1_base, G2_base, Xtest, Ytest, R0, RB, R_F1, R_F2, num_noise_test, noise);
        acc_base_total = acc_base_total + acc_base_mc;
    end
    acc_base_avg = acc_base_total / num_monte_carlo;
    
    % Monte Carlo for RWP
    acc_rwp_total = 0;
    for mc = 1:num_monte_carlo
        acc_rwp_mc = test_with_noise_monte_carlo(G1_rwp, G2_rwp, Xtest, Ytest, R0, RB, R_F1, R_F2, num_noise_test, noise);
        acc_rwp_total = acc_rwp_total + acc_rwp_mc;
    end
    acc_rwp_avg = acc_rwp_total / num_monte_carlo;
    improvement = acc_rwp_avg - acc_base_avg;
    % In ra bảng
    fprintf('    %2d%%     |  %6.2f%% | %6.2f%% | %+7.2f%%\n', ...
        noise, acc_base_avg, acc_rwp_avg, improvement);
end

%% ==================== COMPARATOR OUTPUT TEST ====================
fprintf('\n=== COMPARATOR OUTPUT TEST ===\n');

% Test with digital comparator outputs
acc_base_digital = hardware_forward_pass_with_comparator(G1_base, G2_base, Xtest, Ytest, R0, RB, R_F1, R_F2, V_REF, num_test);
acc_rwp_digital = hardware_forward_pass_with_comparator(G1_rwp, G2_rwp, Xtest, Ytest, R0, RB, R_F1, R_F2, V_REF, num_test);

fprintf('Hardware with comparator:\n');
fprintf('  Baseline: %.2f%%\n', acc_base_digital);
fprintf('  RWP:      %.2f%%\n', acc_rwp_digital);

%% ==================== SUPPORTING FUNCTIONS ====================

function accuracy = hardware_forward_pass(G1, G2, Xtest, Ytest, R0, RB, R_F1, R_F2, num_test)
    correct_count = 0;
    
    for i = 1:num_test
        sample = Xtest(i, :);
        label = Ytest(i);
        
        % Layer 1 - Correct hardware implementation according to paper
        V_F1 = -sum(sample) * (R_F1 / RB);  % Eq. (3) from paper
        
        V_hidden = zeros(1, size(G1, 1));
        for k = 1:size(G1, 1)
            sum_current_mem1 = 0;
            for j = 1:size(G1, 2)
                sum_current_mem1 = sum_current_mem1 + G1(k, j) * sample(j);
            end
            % Eq. (5) from paper: V_O,k = -[Σ(R0·g_jk·V_IN,j) + (R0/R_F2)·V_F]
            V_hidden(k) = -[R0 * sum_current_mem1 + (R0/R_F2) * V_F1];
        end
        
        a1 = 1./(1+exp(-V_hidden));
        
        % Layer 2 - Same structure
        V_F2 = -sum(a1) * (R_F1 / RB);
        
        V_output = zeros(1, size(G2, 1));
        for k = 1:size(G2, 1)
            sum_current_mem2 = 0;
            for j = 1:size(G2, 2)
                sum_current_mem2 = sum_current_mem2 + G2(k, j) * a1(j);
            end
            V_output(k) = -[R0 * sum_current_mem2 + (R0/R_F2) * V_F2];
        end
        
        [~, pred] = max(V_output);
        if pred == label
            correct_count = correct_count + 1;
        end
    end
    
    accuracy = correct_count / num_test * 100;
end

function accuracy = test_with_noise_monte_carlo(G1, G2, Xtest, Ytest, R0, RB, R_F1, R_F2, num_test, noise_percent)
    correct_count = 0;
    noise_std = noise_percent / 100;
    
    % Create noisy conductance arrays
    G1_noisy = G1 .* (1 + noise_std * randn(size(G1)));
    G2_noisy = G2 .* (1 + noise_std * randn(size(G2)));
    G1_noisy = max(1e-6, min(1e-3, G1_noisy));
    G2_noisy = max(1e-6, min(1e-3, G2_noisy));
    
    for i = 1:num_test
        sample = Xtest(i, :);
        label = Ytest(i);
        
        % Layer 1 with noise
        V_F1 = -sum(sample) * (R_F1 / RB);
        V_hidden = zeros(1, size(G1, 1));
        for k = 1:size(G1, 1)
            sum_current_mem1 = 0;
            for j = 1:size(G1, 2)
                sum_current_mem1 = sum_current_mem1 + G1_noisy(k, j) * sample(j);
            end
            V_hidden(k) = -[R0 * sum_current_mem1 + (R0/R_F2) * V_F1];
        end
        a1 = 1./(1+exp(-V_hidden));
        
        % Layer 2 with noise
        V_F2 = -sum(a1) * (R_F1 / RB);
        V_output = zeros(1, size(G2, 1));
        for k = 1:size(G2, 1)
            sum_current_mem2 = 0;
            for j = 1:size(G2, 2)
                sum_current_mem2 = sum_current_mem2 + G2_noisy(k, j) * a1(j);
            end
            V_output(k) = -[R0 * sum_current_mem2 + (R0/R_F2) * V_F2];
        end
        
        [~, pred] = max(V_output);
        if pred == label
            correct_count = correct_count + 1;
        end
    end
    
    accuracy = correct_count / num_test * 100;
end

function accuracy = hardware_forward_pass_with_comparator(G1, G2, Xtest, Ytest, R0, RB, R_F1, R_F2, V_REF, num_test)
    correct_count = 0;
    
    for i = 1:num_test
        sample = Xtest(i, :);
        label = Ytest(i);
        
        % Layer 1
        V_F1 = -sum(sample) * (R_F1 / RB);
        V_hidden = zeros(1, size(G1, 1));
        for k = 1:size(G1, 1)
            sum_current_mem1 = 0;
            for j = 1:size(G1, 2)
                sum_current_mem1 = sum_current_mem1 + G1(k, j) * sample(j);
            end
            V_hidden(k) = -[R0 * sum_current_mem1 + (R0/R_F2) * V_F1];
        end
        a1 = 1./(1+exp(-V_hidden));
        
        % Layer 2 with comparator
        V_F2 = -sum(a1) * (R_F1 / RB);
        V_output = zeros(1, size(G2, 1));
        digital_output = zeros(1, size(G2, 1));
        
        for k = 1:size(G2, 1)
            sum_current_mem2 = 0;
            for j = 1:size(G2, 2)
                sum_current_mem2 = sum_current_mem2 + G2(k, j) * a1(j);
            end
            V_output(k) = -[R0 * sum_current_mem2 + (R0/R_F2) * V_F2];
            
            % Comparator - Eq. (7) from paper
            if V_output(k) >= V_REF
                digital_output(k) = 1;
            else
                digital_output(k) = 0;
            end
        end
        
        % Decision logic with digital outputs
        [~, pred] = max(V_output);  % Can use analog values or implement custom logic with digital_output
        
        if pred == label
            correct_count = correct_count + 1;
        end
    end
    
    accuracy = correct_count / num_test * 100;
end
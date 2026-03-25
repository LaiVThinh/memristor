clear all; close all; clc;

%% ==================== CHIP-IN-THE-LOOP ANALOG NEUROMORPHIC COMPUTING (GPU OPTIMIZED) ====================
fprintf('=== CHIP-IN-THE-LOOP ANALOG NEUROMORPHIC COMPUTING (GPU OPTIMIZED) ===\n');

% Check GPU availability
if gpuDeviceCount > 0
    gpu = gpuDevice();
    fprintf('GPU detected: %s (%.1f GB VRAM)\n', gpu.Name, gpu.AvailableMemory/1e9);
else
    error('No GPU available. This code requires GPU acceleration.');
end

% Load data
load('mnist_train_data.mat');
load('mnist_test_data.mat');

% Format data và chuyển lên GPU
Xtrain = gpuArray(single(X_train')); 
Xtest = gpuArray(single(Xtest'));     
Ytrain = gpuArray(single(Y_train'));  
Ytest = gpuArray(single(Ytest'));     

Xtrain = Xtrain / 255.0;

if min(Ytrain) == 0, Ytrain = Ytrain + 1; end
if min(Ytest) == 0, Ytest = Ytest + 1; end

% Network architecture 
input_size = 784;
hidden_size = 512;
output_size = 10;

% Khởi tạo weights trên GPU - He initialization với single precision
W1 = gpuArray(single(randn(hidden_size, input_size) * sqrt(2/input_size)));
W2 = gpuArray(single(randn(output_size, hidden_size) * sqrt(2/hidden_size)));

fprintf('Network: %d-%d-%d\n', input_size, hidden_size, output_size);
fprintf('Training from scratch with analog chip-in-the-loop (GPU Optimized)\n');

%% ==================== HARDWARE PARAMETERS ====================
G_min = single(1e-5);    G_max = single(1e-4);
G_mid = (G_min + G_max) / 2;

max_weight = max([max(abs(W1(:))), max(abs(W2(:)))]);
R0 = single(max_weight / (G_max - G_mid));
RB = single(1 / G_mid);

% Circuit parameters - GIÁ TRỊ CỐ ĐỊNH ĐƠN GIẢN
R_F1 = single(1e3);     R_F2 = single(1e3);
V_REF = single(0.3);    % Reference voltage cố định

fprintf('G_min = %.2e S, G_max = %.2e S\n', G_min, G_max);
fprintf('R0 = %.2e ohms, RB = %.2e ohms\n', R0, RB);
fprintf('R_F1 = %.2e ohms, R_F2 = %.2e ohms\n', R_F1, R_F2);
fprintf('V_REF = %.2f V (fixed)\n', V_REF);

%% ==================== ANALOG CHIP-IN-THE-LOOP TRAINING (GPU OPTIMIZED) ====================
learning_rate = single(0.05);
num_epochs = 300;
batch_size = 32;
num_train_samples = min(60000, size(Xtrain, 1));
num_test = min(10000, size(Xtest, 1));

grad_clip_threshold = single(1.0);
accuracy_history = zeros(num_epochs, 1, 'single');
loss_history = zeros(num_epochs, 1, 'single');

% Pre-allocate GPU memory for better performance
target_onehot_batch = zeros(batch_size, output_size, 'single', 'gpuArray');

for epoch = 1:num_epochs
    fprintf('Epoch %d/%d: ', epoch, num_epochs);
    
    total_loss = 0;
    correct_predictions = 0;
    
    indices = randperm(size(Xtrain, 1));
    indices = indices(1:num_train_samples);
    
    num_batches = floor(num_train_samples/batch_size);
    
    for batch = 1:num_batches
        batch_start = (batch-1)*batch_size + 1;
        batch_end = batch*batch_size;
        batch_indices = indices(batch_start:batch_end);
        current_batch_size = batch_size;
        
        input_batch = Xtrain(batch_indices, :);
        target_labels_batch = Ytrain(batch_indices);
        
        % ========== FAST ONE-HOT ENCODING ON GPU ==========
        target_onehot_batch(:) = 0;
        lin_idx = (1:current_batch_size)' + (target_labels_batch-1) * current_batch_size;
        target_onehot_batch(lin_idx) = 1;
        
        % ========== ANALOG CIRCUIT FORWARD PASS ==========
        [final_output_voltage_batch, hidden_activation_batch] = analog_forward_pass_gpu_optimized(...
            input_batch, W1, W2, R0, RB, R_F1, R_F2, G_min, G_max);
        
        % ========== ERROR CALCULATION ==========
        output_activation_batch = 1./(1+exp(-final_output_voltage_batch));
        E_d_batch = 0.5 * sum((target_onehot_batch - output_activation_batch).^2, 'all');
        total_loss = total_loss + E_d_batch;
        
        % ========== BACKPROPAGATION ==========
        delta_2_batch = (target_onehot_batch - output_activation_batch) .* output_activation_batch .* (1 - output_activation_batch);
        delta_2_batch = max(-grad_clip_threshold, min(grad_clip_threshold, delta_2_batch));
        
        delta_1_batch = (delta_2_batch * W2) .* hidden_activation_batch .* (1 - hidden_activation_batch);
        delta_1_batch = max(-grad_clip_threshold, min(grad_clip_threshold, delta_1_batch));
        
        % Weight updates với learning rate
        dW2 = (delta_2_batch' * hidden_activation_batch) / current_batch_size;
        dW1 = (delta_1_batch' * input_batch) / current_batch_size;
        
        W2 = W2 + learning_rate * dW2;
        W1 = W1 + learning_rate * dW1;
        
        % ========== ANALOG DECISION ĐƠN GIẢN ==========
        [~, preds] = max(final_output_voltage_batch, [], 2);
        
        % Logic V_REF đơn giản (theo bài báo)
        activated_neurons = (final_output_voltage_batch >= V_REF);
        num_activated = sum(activated_neurons, 2);
        
        single_activated = find(num_activated == 1);
        if ~isempty(single_activated)
            [~, single_preds] = max(activated_neurons(single_activated, :), [], 2);
            preds(single_activated) = single_preds;
        end
        
        correct_predictions = correct_predictions + sum(preds == target_labels_batch);
    end
    
    % Process remaining samples if any
    remaining_samples = mod(num_train_samples, batch_size);
    if remaining_samples > 0
        batch_indices = indices(end-remaining_samples+1:end);
        input_batch = Xtrain(batch_indices, :);
        target_labels_batch = Ytrain(batch_indices);
        
        target_onehot_small = zeros(remaining_samples, output_size, 'single', 'gpuArray');
        lin_idx = (1:remaining_samples)' + (target_labels_batch-1) * remaining_samples;
        target_onehot_small(lin_idx) = 1;
        
        [final_output_voltage_batch, hidden_activation_batch] = analog_forward_pass_gpu_optimized(...
            input_batch, W1, W2, R0, RB, R_F1, R_F2, G_min, G_max);
        
        output_activation_batch = 1./(1+exp(-final_output_voltage_batch));
        E_d_batch = 0.5 * sum((target_onehot_small - output_activation_batch).^2, 'all');
        total_loss = total_loss + E_d_batch;
        
        [~, preds] = max(final_output_voltage_batch, [], 2);
        correct_predictions = correct_predictions + sum(preds == target_labels_batch);
    end
    
    epoch_accuracy = gather(correct_predictions) / num_train_samples * 100;
    epoch_avg_loss = gather(total_loss) / num_train_samples;
    
    accuracy_history(epoch) = epoch_accuracy;
    loss_history(epoch) = epoch_avg_loss;
    
    % Test accuracy
    test_acc = test_analog_gpu_optimized(W1, W2, Xtest, Ytest, num_test, R0, RB, R_F1, R_F2, G_min, G_max, V_REF);
    
    fprintf(' Loss: %.4f, Train Acc: %.2f%%, Test Acc: %.2f%%\n', ...
            epoch_avg_loss, epoch_accuracy, test_acc);
    
    % Adaptive learning rate đơn giản
    if epoch > 10 && loss_history(epoch) > loss_history(epoch-1)
        learning_rate = learning_rate * single(0.95);
        fprintf('  Learning rate reduced to %.6f\n', learning_rate);
    end
    
    % Early stopping check
    if epoch > 20 && accuracy_history(epoch) < accuracy_history(epoch-5)
        fprintf('Early stopping at epoch %d\n', epoch);
        break;
    end
end

%% ==================== FINAL RESULTS ====================
fprintf('\n=== FINAL ANALOG CIRCUIT RESULTS ===\n');

% Gather results to CPU
W1_cpu = gather(W1); W2_cpu = gather(W2);
Xtest_cpu = gather(Xtest); Ytest_cpu = gather(Ytest);

final_test_acc = test_analog_gpu_optimized(W1_cpu, W2_cpu, Xtest_cpu, Ytest_cpu, ...
    min(5000, size(Xtest_cpu,1)), R0, RB, R_F1, R_F2, G_min, G_max, V_REF);

fprintf('Final Analog Test Accuracy: %.2f%%\n', final_test_acc);
fprintf('Final V_REF: %.2f V (fixed)\n', V_REF);
fprintf('Training Progress: %.2f%% -> %.2f%%\n', accuracy_history(1), accuracy_history(end));

% Analyze final conductance distribution
G1_final = 1/RB - W1_cpu/R0; G1_final = max(G_min, min(G_max, G1_final));
G2_final = 1/RB - W2_cpu/R0; G2_final = max(G_min, min(G_max, G2_final));

% Power estimation
avg_conductance = mean([G1_final(:); G2_final(:)]);
V_supply = single(1.0);
power_estimate = avg_conductance * V_supply^2 * (input_size * hidden_size + hidden_size * output_size);
fprintf('Estimated Analog Power: ~%.2f mW\n', power_estimate * 1000);

%% ==================== GPU OPTIMIZED FUNCTIONS ====================

function [analog_out_batch, hidden_act_batch] = analog_forward_pass_gpu_optimized(input_batch, W1, W2, R0, RB, R_F1, R_F2, G_min, G_max)
    % Convert weights to conductance - Equation (6) from paper
    G1 = 1/RB - W1/R0;
    G2 = 1/RB - W2/R0;
    G1 = max(G_min, min(G_max, G1));
    G2 = max(G_min, min(G_max, G2));
    
    % ========== LAYER 1: Single Crossbar Array + Constant-Term Circuit ==========
    V_F1_batch = -sum(input_batch, 2) * (R_F1 / RB);
    
    current_mem1 = input_batch * G1';
    current_fb1 = V_F1_batch / R_F2;
    total_current1 = current_mem1 + current_fb1;
    
    hidden_voltage_batch = -R0 * total_current1;
    hidden_act_batch = 1./(1+exp(-hidden_voltage_batch));
    
    % ========== LAYER 2: Same Circuit Architecture ==========
    V_F2_batch = -sum(hidden_act_batch, 2) * (R_F1 / RB);
    current_mem2 = hidden_act_batch * G2';
    current_fb2 = V_F2_batch / R_F2;
    total_current2 = current_mem2 + current_fb2;
    
    analog_out_batch = -R0 * total_current2;
end

function accuracy = test_analog_gpu_optimized(W1, W2, Xtest, Ytest, num_test, R0, RB, R_F1, R_F2, G_min, G_max, V_REF)
    num_test = min(floor(num_test), size(Xtest, 1));
    
    [final_output_voltage_batch, ~] = analog_forward_pass_gpu_optimized(...
        Xtest(1:num_test,:), W1, W2, R0, RB, R_F1, R_F2, G_min, G_max);
    
    % Vectorized decision logic với V_REF cố định
    [~, preds] = max(final_output_voltage_batch, [], 2);
    activated_neurons = (final_output_voltage_batch >= V_REF);
    num_activated = sum(activated_neurons, 2);
    
    single_activated = find(num_activated == 1);
    if ~isempty(single_activated)
        [~, single_preds] = max(activated_neurons(single_activated, :), [], 2);
        preds(single_activated) = single_preds;
    end
    
    correct_count = sum(preds == Ytest(1:num_test));
    accuracy = gather(correct_count) / num_test * 100;
end
// --- Kernels for Matrix Multiplication ---
__kernel void matmul_forward(__global const float* A, __global const float* B, __global float* C, int M, int K, int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k_loop = 0; k_loop < K; k_loop++) // Correção: Adicionado condição e incremento
    {
        float val_a = A[row * K + k_loop];
        float val_b = B[k_loop * N + col];
        sum += val_a * val_b;
    }
    C[row * N + col] = sum;
}

__kernel void matmul_backward_input(
    __global const float* dL_dC,
    __global const float* B_T,
    __global float* dL_dA,
    int M, int N_dim, int K_dim)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < K_dim) {
        float sum = 0.0f;
        for (int i = 0; i < N_dim; i++) {
            sum += dL_dC[row * N_dim + i] * B_T[i * K_dim + col];
        }
        dL_dA[row * K_dim + col] = sum;
    }
}

__kernel void matmul_backward_weights_accumulate(
    __global const float* A_T,
    __global const float* dL_dC,
    __global float* dL_dB,
    int K_dim, int M_dim, int N_dim)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < K_dim && col < N_dim) {
        float sum = 0.0f;
        for (int i = 0; i < M_dim; i++) {
            sum += A_T[row * M_dim + i] * dL_dC[i * N_dim + col];
        }
        dL_dB[row * N_dim + col] += sum;
    }
}

// --- Kernels for Activation Functions ---
__kernel void sigmoid_forward(
    __global const float* input_arr,
    __global float* output_arr,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        output_arr[gid] = 1.0f / (1.0f + exp(-input_arr[gid]));
    }
}

__kernel void tanh_forward(
    __global const float* input_arr,
    __global float* output_arr,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        output_arr[gid] = tanh(input_arr[gid]);
    }
}

__kernel void relu_forward(
    __global const float* input_arr,
    __global float* output_arr,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        output_arr[gid] = (input_arr[gid] > 0.0f) ? input_arr[gid] : 0.0f;
    }
}

__kernel void relu_backward(
    __global const float* dL_dy,
    __global const float* input_pre_relu,
    __global float* dL_dx,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        dL_dx[gid] = dL_dy[gid] * ((input_pre_relu[gid] > 0.0f) ? 1.0f : 0.0f);
    }
}

// --- Kernel for Layer Normalization ---
__kernel void layernorm_forward(
    __global const float* x,
    __global const float* gamma,
    __global const float* beta,
    __global float* output_ln,
    __global float* means_out,
    __global float* inv_stds_out,
    int batch_size,
    int seq_len,
    int embedding_dim,
    float epsilon,
    __global int* debug_flag_nan)
{
    int token_idx_flat = get_global_id(0);
    if (token_idx_flat < batch_size * seq_len) {
        float sum = 0.0f;
        int base_addr_x = token_idx_flat * embedding_dim;

        for (int d = 0; d < embedding_dim; d++) {
            float val_x = x[base_addr_x + d];
            if (!isfinite(val_x)) { atomic_or(debug_flag_nan, 1024); }
            sum += val_x;
        }
        float mean = sum / (float)embedding_dim;
        if (!isfinite(mean)) { atomic_or(debug_flag_nan, 1); return; }

        float variance_sum = 0.0f;
        for (int d = 0; d < embedding_dim; d++) {
            float diff = x[base_addr_x + d] - mean;
            variance_sum += diff * diff;
        }
        float variance = variance_sum / (float)embedding_dim;

        if (!isfinite(variance)) { atomic_or(debug_flag_nan, 2); return; }
        if (variance < 0.0f) { atomic_or(debug_flag_nan, 2048); }
        if ((variance + epsilon) == 0.0f) { atomic_or(debug_flag_nan, 4096); }

        float inv_std = rsqrt(variance + epsilon);
        if (!isfinite(inv_std)) { atomic_or(debug_flag_nan, 4); }

        for (int d = 0; d < embedding_dim; d++) {
            float normalized_x = (x[base_addr_x + d] - mean) * inv_std;
            float val_out = gamma[d] * normalized_x + beta[d];
            if (!isfinite(val_out)) {
                atomic_or(debug_flag_nan, 8);
                if (!isfinite(gamma[d])) atomic_or(debug_flag_nan, 512);
                if (!isfinite(beta[d])) atomic_or(debug_flag_nan, 256);
                if (!isfinite(normalized_x)) atomic_or(debug_flag_nan, 128);
            }
            output_ln[base_addr_x + d] = val_out;
        }

        means_out[token_idx_flat] = mean;
        inv_stds_out[token_idx_flat] = inv_std;
    }
}

__kernel void layernorm_backward(
    __global const float* dL_dy,
    __global const float* x_input,
    __global const float* mean_vals,
    __global const float* inv_std_vals,
    __global const float* gamma_data,
    __global float* dL_dx,
    __global float* dL_dgamma_accum,
    __global float* dL_dbeta_accum,
    int batch_size,
    int seq_len,
    int embedding_dim,
    float epsilon)
{
    int batch_idx = get_global_id(0);
    int seq_idx = get_global_id(1);

    if (batch_idx < batch_size && seq_idx < seq_len) {
        int mean_inv_std_flat_idx = batch_idx * seq_len + seq_idx;
        float mean = mean_vals[mean_inv_std_flat_idx];
        float inv_std = inv_std_vals[mean_inv_std_flat_idx];

        float sum_dL_dy_gamma_scaled_slice = 0.0f;
        float sum_dL_dy_gamma_x_minus_mean_slice = 0.0f;

        for (int d = 0; d < embedding_dim; d++) {
            int current_flat_idx = (batch_idx * seq_len + seq_idx) * embedding_dim + d;
            float dL_dy_val = dL_dy[current_flat_idx];
            float x_val = x_input[current_flat_idx];
            float gamma_val_d = gamma_data[d];

            float dL_dx_scaled_val_d = dL_dy_val * gamma_val_d;
            sum_dL_dy_gamma_scaled_slice += dL_dx_scaled_val_d;
            sum_dL_dy_gamma_x_minus_mean_slice += dL_dx_scaled_val_d * (x_val - mean);
        }

        float dL_dVar_slice = sum_dL_dy_gamma_x_minus_mean_slice * (-0.5f * pow(inv_std, 3.0f));
        float dL_dMean_slice = -inv_std * sum_dL_dy_gamma_scaled_slice;

        for (int d = 0; d < embedding_dim; d++) {
            int current_flat_idx = (batch_idx * seq_len + seq_idx) * embedding_dim + d;
            float dL_dy_val = dL_dy[current_flat_idx];
            float x_val = x_input[current_flat_idx];
            float gamma_val_d = gamma_data[d];

            float dL_dx_scaled_val_d = dL_dy_val * gamma_val_d;
            float term1 = dL_dx_scaled_val_d * inv_std;
            float term2 = dL_dVar_slice * (2.0f * (x_val - mean) / (float)embedding_dim);
            float term3 = dL_dMean_slice / (float)embedding_dim;
            dL_dx[current_flat_idx] = term1 + term2 + term3;

            dL_dgamma_accum[current_flat_idx] = dL_dy_val * (x_val - mean) * inv_std;
            dL_dbeta_accum[current_flat_idx] = dL_dy_val;
        }
    }
}

// --- Kernels for Softmax ---
__kernel void softmax_forward(
    __global const float* scores,
    __global float* probs,
    int M,
    int N,
    __global float* work_buffer_max_scores,
    __global float* work_buffer_sum_exp,
    __global int* debug_flag_nan)
{
    int row = get_global_id(0);
    if (row < M) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < N; col++) {
            if (scores[row * N + col] > max_val) {
                max_val = scores[row * N + col];
            }
        }
        if (!isfinite(max_val)) { atomic_or(debug_flag_nan, 16); }

        float sum_exp = 0.0f;
        for (int col = 0; col < N; col++) {
            float current_score = scores[row * N + col];
            if (!isfinite(current_score)) { atomic_or(debug_flag_nan, 2048); }

            float val_to_exp = current_score - max_val;
            float exp_val = exp(val_to_exp);
            if (!isfinite(exp_val)) { atomic_or(debug_flag_nan, 32); }

            probs[row * N + col] = exp_val;
            sum_exp += exp_val;
        }

        if (!isfinite(sum_exp)) { atomic_or(debug_flag_nan, 64); }
        if (sum_exp == 0.0f) { atomic_or(debug_flag_nan, 128); }

        for (int col = 0; col < N; col++) {
            float final_prob = (sum_exp == 0.0f) ? (1.0f / (float)N) : (probs[row * N + col] / sum_exp);
            if (!isfinite(final_prob)) { atomic_or(debug_flag_nan, 256); }
            probs[row * N + col] = final_prob;
        }
    }
}

__kernel void softmax_backward(
    __global const float* dL_dprobs,
    __global const float* probs,
    __global float* dL_dscores,
    int M, int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum_dL_dP_P = 0.0f;
        for (int k = 0; k < N; k++) {
            sum_dL_dP_P += dL_dprobs[row * N + k] * probs[row * N + k];
        }
        dL_dscores[row * N + col] = probs[row * N + col] * (dL_dprobs[row * N + col] - sum_dL_dP_P);
    }
}

// --- Kernels for Element-wise Operations ---
__kernel void elementwise_add_forward(
    __global const float* A_in,
    __global const float* B_in,
    __global float* C_out,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        C_out[gid] = A_in[gid] + B_in[gid];
    }
}

__kernel void elementwise_add_backward_accumulate(
    __global float* dL_dA,
    __global const float* dL_dC,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        dL_dA[gid] += dL_dC[gid];
    }
}

__kernel void elementwise_multiply(
    __global const float* A_in,
    __global const float* B_in,
    __global float* C_out,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        C_out[gid] = A_in[gid] * B_in[gid];
    }
}

// --- Kernels for Buffer Manipulation ---
__kernel void copy_buffer(
    __global const float* source_buf,
    __global float* destination_buf,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        destination_buf[gid] = source_buf[gid];
    }
}

__kernel void transpose(
    __global const float* A,
    __global float* B,
    int M,
    int N)
{
    int row_B = get_global_id(0);
    int col_B = get_global_id(1);
    if (row_B < N && col_B < M) {
        B[row_B * M + col_B] = A[col_B * N + row_B];
    }
}

__kernel void scale_and_mask_scores(
    __global float* scores,
    float scale_factor,
    __global const float* mask,
    int apply_mask,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        scores[gid] *= scale_factor;
        if (apply_mask == 1 && mask[gid] < 0.0f) {
            scores[gid] = -1.0e30f;
        }
    }
}

__kernel void scale_buffer(
    __global float* buffer_in_out,
    float scalar,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        buffer_in_out[gid] *= scalar;
    }
}

// --- Kernel de Teste ---
__kernel void test_gpu(
    __global float* buffer,
    int size)
{
    int idx = get_global_id(0);
    if (idx < size) {
        buffer[idx] = idx * 1.0f;
    }
}
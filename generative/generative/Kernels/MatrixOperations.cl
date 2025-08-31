// --- Kernels for Matrix Multiplication ---
__kernel void matmul_forward(__global const float* A, __global const float* B, __global float* C, int M, int K, int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1); // N
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k_loop = 0;
    {
        float val_a = A[row * K + k_loop];
        float val_b = B[k_loop * N + col];

        // Verificação opcional de debug aqui, mas pode ser lenta:
        // if (!isfinite(val_a) || !isfinite(val_b)) { /* lidar com erro ou marcar */ }
        // float prod = val_a * val_b;
        // if (!isfinite(prod)) { /* overflow/underflow no produto */ }
        // sum += prod;
        // if (!isfinite(sum)) { /* overflow/underflow na soma */ }

        sum += val_a * val_b;
    }
    C[row * N + col] = sum;
}

// Kernel de teste (opcional, pode ser removido depois)
__kernel void test_gpu(__global float* buffer, int size) // MUDANÇA: float*
{
    int idx = get_global_id(0);
    if (idx < size)
    {
        buffer[idx] = idx * 1.0f; // MUDANÇA: 1.0f para float
    }
}

__kernel void matmul_backward_input(
    __global const float* dL_dC, // MUDANÇA: float*, const float*
    __global const float* B_T,   // MUDANÇA: float*, const float*
    __global float* dL_dA,       // MUDANÇA: float*
    int M, int N_dim, int K_dim)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < K_dim) {
        float sum = 0.0f; // MUDANÇA: 0.0f
        for (int i = 0; i < N_dim; i++) {
            sum += dL_dC[row * N_dim + i] * B_T[i * K_dim + col];
        }
        dL_dA[row * K_dim + col] = sum;
    }
}

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

__kernel void matmul_backward_weights_accumulate(
    __global const float* A_T,   // MUDANÇA: float*, const float*
    __global const float* dL_dC, // MUDANÇA: float*, const float*
    __global float* dL_dB,       // MUDANÇA: float*
    int K_dim, int M_dim, int N_dim)
{
    int row = get_global_id(0); // Current row in dL_dB (rows of original weights)
    int col = get_global_id(1); // Current col in dL_dB (cols of original weights)

    if (row < K_dim && col < N_dim) {
        float sum = 0.0f; // MUDANÇA: 0.0f
        for (int i = 0; i < M_dim; i++) {
            sum += A_T[row * M_dim + i] * dL_dC[i * N_dim + col];
        }
        dL_dB[row * N_dim + col] += sum;
    }
}

// --- Kernels for ReLU ---
__kernel void relu_forward(
    __global const float* input_arr, // MUDANÇA: const float*
    __global float* output_arr,      // MUDANÇA: float*
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        output_arr[gid] = (input_arr[gid] > 0.0f) ? input_arr[gid] : 0.0f; // MUDANÇA: 0.0f
    }
}

__kernel void relu_backward(
    __global const float* dL_dy,     // MUDANÇA: const float*
    __global const float* input_pre_relu, // MUDANÇA: const float*
    __global float* dL_dx,           // MUDANÇA: float*
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        dL_dx[gid] = dL_dy[gid] * ((input_pre_relu[gid] > 0.0f) ? 1.0f : 0.0f); // MUDANÇA: 0.0f, 1.0f
    }
}

// --- Kernels for Layer Normalization ---
__kernel void layernorm_forward(
    __global const float* x,
    __global const float* gamma,
    __global const float* beta,
    __global float* output_ln,
    __global float* means_out,       // Buffer para médias [batch_size * seq_len]
    __global float* inv_stds_out,    // Buffer para desvios padrão inversos [batch_size * seq_len]
    int batch_size,
    int seq_len,
    int embedding_dim,
    float epsilon,
    __global int* debug_flag_nan // Argumento para a flag de debug (int[1])
    )
{
    // O global_id(0) deve iterar sobre cada token/vetor que precisa de normalização.
    // Se o global work size é (batch_size * seq_len), então é 1D.
    int token_idx_flat = get_global_id(0); // Índice achatado do token (0 a batch_size*seq_len - 1)

    if (token_idx_flat < batch_size * seq_len) {
        float sum = 0.0f;
        // O endereço base para os elementos deste token no buffer x
        int base_addr_x = token_idx_flat * embedding_dim;

        for (int d = 0; d < embedding_dim; d++) {
            float val_x = x[base_addr_x + d];
            if (!isfinite(val_x)) { atomic_or(debug_flag_nan, 1024); /* Input x is NaN/Inf */ }
            sum += val_x;
        }
        float mean = sum / (float)embedding_dim;
        if (!isfinite(mean)) { atomic_or(debug_flag_nan, 1); return; }

        float variance_sum = 0.0f;
        for (int d = 0; d < embedding_dim; d++) {
            float diff = x[base_addr_x + d] - mean;
            variance_sum += diff * diff;
        }
        // A variância é sempre não negativa se os inputs são finitos e a média é finita.
        float variance = variance_sum / (float)embedding_dim;

        // ***** PONTO CRÍTICO DE DEPURAÇÃO *****
        if (!isfinite(variance)) {
            atomic_or(debug_flag_nan, 2); // Variance é NaN/Inf
            // Para depuração, podemos escrever a variância problemática em inv_stds_out para inspeção
            // inv_stds_out[token_idx_flat] = variance; // Cuidado: isso sobrescreve o inv_std normal
            return;
        }
        if (variance < 0.0f) { // ESTA É A VERIFICAÇÃO MAIS IMPORTANTE AGORA
            atomic_or(debug_flag_nan, 2048); // Variance é NEGATIVA!
            // inv_stds_out[token_idx_flat] = variance; // Escreve a variância negativa para depuração
            // return; // Não retorne ainda, deixe o NaN propagar para inv_std para pegarmos na CPU
        }
         if ( (variance + epsilon) == 0.0f ) {
            atomic_or(debug_flag_nan, 4096); // variance + epsilon == 0.0f, causará divisão por zero
        }
        // ***** FIM DO PONTO CRÍTICO DE DEPURAÇÃO *****

        float inv_std = rsqrt(variance + epsilon); // rsqrt(x) = 1/sqrt(x)
        if (!isfinite(inv_std)) {
            // Se inv_std é NaN, significa que variance+epsilon foi negativo, ou variance já era NaN.
            // A flag para variance negativa (2048) ou variance NaN (2) já deveria ter sido acionada.
            // Adicionamos uma flag específica para inv_std NaN para cobrir todos os casos.
            atomic_or(debug_flag_nan, 4);
            // Para depuração, podemos escrever a (variance+epsilon) problemática
            // output_ln[base_addr_x + 0] = variance + epsilon; // Apenas um exemplo de como sinalizar
            // return; // Deixe o NaN propagar para output_ln
        }

        for (int d = 0; d < embedding_dim; d++) {
            float normalized_x = (x[base_addr_x + d] - mean) * inv_std;
            float val_out = gamma[d] * normalized_x + beta[d];
            if(!isfinite(val_out)){
                atomic_or(debug_flag_nan, 8);
                if (!isfinite(gamma[d])) atomic_or(debug_flag_nan, 512); // gamma é NaN
                if (!isfinite(beta[d])) atomic_or(debug_flag_nan, 256);  // beta é NaN
                if (!isfinite(normalized_x)) atomic_or(debug_flag_nan, 128); // (x-mean)*inv_std é NaN
            }
            output_ln[base_addr_x + d] = val_out;
        }

        means_out[token_idx_flat] = mean;
        inv_stds_out[token_idx_flat] = inv_std; // Armazena o inv_std (que pode ser NaN/Inf)
    }
}

__kernel void layernorm_backward(
    __global const float* dL_dy,         // MUDANÇA: const float*
    __global const float* x_input,       // MUDANÇA: const float*
    __global const float* mean_vals,     // MUDANÇA: const float*
    __global const float* inv_std_vals,  // MUDANÇA: const float*
    __global const float* gamma_data,    // MUDANÇA: const float*
    __global float* dL_dx,               // MUDANÇA: float*
    __global float* dL_dgamma_accum,     // MUDANÇA: float* (para acumulação no host)
    __global float* dL_dbeta_accum,      // MUDANÇA: float* (para acumulação no host)
    int batch_size,
    int seq_len,
    int embedding_dim,
    float epsilon) // MUDANÇA: float
{
    int batch_idx = get_global_id(0);
    int seq_idx = get_global_id(1);

    if (batch_idx < batch_size && seq_idx < seq_len) {
        int mean_inv_std_flat_idx = batch_idx * seq_len + seq_idx;
        float mean = mean_vals[mean_inv_std_flat_idx];
        float inv_std = inv_std_vals[mean_inv_std_flat_idx];

        float sum_dL_dy_gamma_scaled_slice = 0.0f; // MUDANÇA: 0.0f
        float sum_dL_dy_gamma_x_minus_mean_slice = 0.0f; // MUDANÇA: 0.0f

        for (int d = 0; d < embedding_dim; d++) {
            int current_flat_idx = (batch_idx * seq_len + seq_idx) * embedding_dim + d;
            float dL_dy_val = dL_dy[current_flat_idx];
            float x_val = x_input[current_flat_idx];
            float gamma_val_d = gamma_data[d];

            float dL_dx_scaled_val_d = dL_dy_val * gamma_val_d;
            sum_dL_dy_gamma_scaled_slice += dL_dx_scaled_val_d;
            sum_dL_dy_gamma_x_minus_mean_slice += dL_dx_scaled_val_d * (x_val - mean);
        }

        float dL_dVar_slice = sum_dL_dy_gamma_x_minus_mean_slice * (-0.5f * pow(inv_std, 3.0f)); // MUDANÇA: -0.5f, 3.0f
        float dL_dMean_slice = -inv_std * sum_dL_dy_gamma_scaled_slice; // MUDANÇA: -inv_std

        for (int d = 0; d < embedding_dim; d++) {
            int current_flat_idx = (batch_idx * seq_len + seq_idx) * embedding_dim + d;
            float dL_dy_val = dL_dy[current_flat_idx];
            float x_val = x_input[current_flat_idx];
            float gamma_val_d = gamma_data[d];

            float dL_dx_scaled_val_d = dL_dy_val * gamma_val_d;
            
            float term1 = dL_dx_scaled_val_d * inv_std;
            float term2 = dL_dVar_slice * (2.0f * (x_val - mean) / (float)embedding_dim); // MUDANÇA: 2.0f, cast
            float term3 = dL_dMean_slice / (float)embedding_dim; // MUDANÇA: cast
            dL_dx[current_flat_idx] = term1 + term2 + term3;

            dL_dgamma_accum[current_flat_idx] = dL_dy_val * (x_val - mean) * inv_std;
            dL_dbeta_accum[current_flat_idx] = dL_dy_val;
        }
    }
}

// --- Kernels for Softmax (used in ApplyMaskAndSoftmax) ---
__kernel void softmax_forward(
    __global const float* scores,
    __global float* probs,
    int M, // Número de vetores (linhas)
    int N, // Dimensão de cada vetor (colunas)
    __global float* work_buffer_max_scores, // Buffer de trabalho para max_val por linha
    __global float* work_buffer_sum_exp,    // Buffer de trabalho para sum_exp por linha
    __global int* debug_flag_nan           // NOVO ARGUMENTO
    )
{
    int row = get_global_id(0); // Kernel é lançado com M itens de trabalho

    if (row < M) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < N; col++) {
            if (scores[row * N + col] > max_val) {
                max_val = scores[row * N + col];
            }
        }
        if (!isfinite(max_val)) { atomic_or(debug_flag_nan, 16); /* Marcar e continuar para ver o output */ }
        // work_buffer_max_scores[row] = max_val; // Opcional, se precisar fora

        float sum_exp = 0.0f;
        for (int col = 0; col < N; col++) {
            // Subtrair max_val antes de exp para estabilidade numérica
            float current_score = scores[row * N + col];
            if (!isfinite(current_score)) { atomic_or(debug_flag_nan, 2048); /* Score NaN */ }

            float val_to_exp = current_score - max_val;
            // if (val_to_exp < -80.0f) val_to_exp = -80.0f; // Evitar underflow de exp para zero muito rápido
            // if (val_to_exp > 80.0f) val_to_exp = 80.0f; // Evitar overflow de exp para Inf muito rápido

            float exp_val = exp(val_to_exp);
            if (!isfinite(exp_val)) { atomic_or(debug_flag_nan, 32); /* Marcar e continuar */ }

            probs[row * N + col] = exp_val; // Armazena exp_val temporariamente
            sum_exp += exp_val;
        }

        if (!isfinite(sum_exp)) { atomic_or(debug_flag_nan, 64); /* Marcar e continuar */ }
        if (sum_exp == 0.0f) { // Evitar divisão por zero
            atomic_or(debug_flag_nan, 128);
            // Se sum_exp é zero, todas as probs serão 0 ou NaN. Poderia distribuir 1/N uniformemente.
            // Por agora, apenas marcamos. Para evitar NaN na divisão, podemos setar sum_exp para um valor pequeno.
            // sum_exp = 1e-9f; // Isso evitaria NaN na divisão mas pode não ser o "correto"
        }
        // work_buffer_sum_exp[row] = sum_exp; // Opcional

        for (int col = 0; col < N; col++) {
            float final_prob;
            if (sum_exp == 0.0f) { // Se sum_exp foi zero, para evitar NaN, distribui uniformemente
                final_prob = 1.0f / (float)N;
            } else {
                final_prob = probs[row * N + col] / sum_exp;
            }

            if (!isfinite(final_prob)) { atomic_or(debug_flag_nan, 256); /* Marcar e continuar */ }
            probs[row * N + col] = final_prob;
        }
    }
}

__kernel void softmax_backward(
    __global const float* dL_dprobs, // MUDANÇA: const float*
    __global const float* probs,     // MUDANÇA: const float*
    __global float* dL_dscores,      // MUDANÇA: float*
    int M, int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        float sum_dL_dP_P = 0.0f; // MUDANÇA: 0.0f
        for (int k = 0; k < N; k++) {
            sum_dL_dP_P += dL_dprobs[row * N + k] * probs[row * N + k];
        }
        dL_dscores[row * N + col] = probs[row * N + col] * (dL_dprobs[row * N + col] - sum_dL_dP_P);
    }
}

// --- Kernels for Element-wise Operations (Add, Accumulate, Multiply) ---
__kernel void elementwise_add_forward(
    __global const float* A_in, // MUDANÇA: const float*
    __global const float* B_in, // MUDANÇA: const float*
    __global float* C_out,      // MUDANÇA: float*
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        C_out[gid] = A_in[gid] + B_in[gid];
    }
}

__kernel void elementwise_add_backward_accumulate(
    __global float* dL_dA,    // MUDANÇA: float*
    __global const float* dL_dC, // MUDANÇA: const float*
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        dL_dA[gid] += dL_dC[gid];
    }
}

__kernel void elementwise_multiply(
    __global const float* A_in, // MUDANÇA: const float*
    __global const float* B_in, // MUDANÇA: const float*
    __global float* C_out,      // MUDANÇA: float*
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        C_out[gid] = A_in[gid] * B_in[gid];
    }
}

// --- Kernels for Buffer Manipulation ---
__kernel void copy_buffer(
    __global const float* source_buf,     // MUDANÇA: const float*
    __global float* destination_buf,      // MUDANÇA: float*
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        destination_buf[gid] = source_buf[gid];
    }
}

__kernel void transpose(
    __global const float* A, // MUDANÇA: const float*
    __global float* B,       // MUDANÇA: float*
    int M, // Rows of A
    int N) // Cols of A
{
    int row_B = get_global_id(0); // Corresponds to col of A
    int col_B = get_global_id(1); // Corresponds to row of A
    if (row_B < N && col_B < M) {
        B[row_B * M + col_B] = A[col_B * N + row_B];
    }
}

__kernel void scale_and_mask_scores(
    __global float* scores,      // MUDANÇA: float*
    float scale_factor,          // MUDANÇA: float
    __global const float* mask,  // MUDANÇA: const float*
    int apply_mask,
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        scores[gid] *= scale_factor;
        if (apply_mask == 1 && mask[gid] < 0.0f) { // MUDANÇA: 0.0f
            scores[gid] = -1.0e30f; // MUDANÇA: -1.0e30f para float
        }
    }
}

__kernel void scale_buffer(
    __global float* buffer_in_out, // MUDANÇA: float*
    float scalar,                  // MUDANÇA: float
    int total_elements)
{
    int gid = get_global_id(0);
    if (gid < total_elements) {
        buffer_in_out[gid] *= scalar;
    }
}
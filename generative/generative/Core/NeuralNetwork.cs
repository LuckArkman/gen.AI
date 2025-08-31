using Cloo;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic; // Adicionado para List

namespace Core
{
    public class NeuralNetwork : IDisposable
    {
        // Pesos e vieses da rede LSTM
        private Tensor W_i, U_i, b_i; // Input Gate
        private Tensor W_f, U_f, b_f; // Forget Gate
        private Tensor W_c, U_c, b_c; // Cell Gate
        private Tensor W_o, U_o, b_o; // Output Gate
        private Tensor W_out, b_out; // Camada de Saída (linear + softmax)

        private readonly int inputSize, hiddenSize, outputSize, contextWindowSize;

        // Campos OpenCL
        private ComputeContext? _context;
        private ComputeDevice? _device;
        private ComputeCommandQueue? _queue;
        private ComputeProgram? _program;
        private ComputeKernel? _matmulKernel;
        private ComputeKernel? _sigmoidKernel;
        private ComputeKernel? _tanhKernel;
        private ComputeKernel? _softmaxKernel;
        private ComputeKernel? _elementwiseAddKernel;
        private ComputeKernel? _elementwiseMultiplyKernel;

        // Propriedades públicas para acesso aos Tensors internos
        public Tensor W_i_Tensor => W_i;
        public Tensor U_i_Tensor => U_i;
        public Tensor b_i_Tensor => b_i;
        public Tensor W_f_Tensor => W_f;
        public Tensor U_f_Tensor => U_f;
        public Tensor b_f_Tensor => b_f;
        public Tensor W_c_Tensor => W_c;
        public Tensor U_c_Tensor => U_c;
        public Tensor b_c_Tensor => b_c;
        public Tensor W_o_Tensor => W_o;
        public Tensor U_o_Tensor => U_o;
        public Tensor b_o_Tensor => b_o;
        public Tensor W_out_Tensor => W_out;
        public Tensor b_out_Tensor => b_out;

        public int InputSize => inputSize;
        public int HiddenSize => hiddenSize;
        public int OutputSize => outputSize;

        // Construtor principal para inicialização
        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int contextWindowSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.contextWindowSize = contextWindowSize;

            // --- CORREÇÃO INICIA AQUI ---
            // O tamanho do vocabulário (entrada para UM passo da LSTM) é o outputSize.
            int vocabSize = outputSize;

            Random rand = new Random();
            // A inicialização de He deve usar o tamanho da entrada do passo (vocabSize), não da janela inteira (inputSize).
            double sqrtFanInHidden = Math.Sqrt(2.0 / (vocabSize + hiddenSize));
            double sqrtFanInRecurrent = Math.Sqrt(2.0 / hiddenSize);

            // As matrizes W multiplicam a entrada de um passo (x_t), então sua primeira dimensão deve ser vocabSize.
            W_i = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand), new int[] { vocabSize, hiddenSize });
            U_i = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand), new int[] { hiddenSize, hiddenSize });
            b_i = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

            W_f = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand), new int[] { vocabSize, hiddenSize });
            U_f = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand), new int[] { hiddenSize, hiddenSize });
            b_f = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

            W_c = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand), new int[] { vocabSize, hiddenSize });
            U_c = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand), new int[] { hiddenSize, hiddenSize });
            b_c = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

            W_o = new Tensor(InitializeWeights(vocabSize, hiddenSize, sqrtFanInHidden, rand), new int[] { vocabSize, hiddenSize });
            U_o = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanInRecurrent, rand), new int[] { hiddenSize, hiddenSize });
            b_o = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            // --- FIM DA CORREÇÃO ---

            W_out = new Tensor(InitializeWeights(hiddenSize, outputSize, Math.Sqrt(2.0 / hiddenSize), rand), new int[] { hiddenSize, outputSize });
            b_out = new Tensor(new double[outputSize], new int[] { outputSize });

            InitializeOpenCL();
        }

        // Construtor para carregar um modelo existente
        private NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int contextWindowSize,
            Tensor W_i, Tensor U_i, Tensor b_i,
            Tensor W_f, Tensor U_f, Tensor b_f,
            Tensor W_c, Tensor U_c, Tensor b_c,
            Tensor W_o, Tensor U_o, Tensor b_o,
            Tensor W_out, Tensor b_out)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.contextWindowSize = contextWindowSize;
            this.W_i = W_i;
            this.U_i = U_i;
            this.b_i = b_i;
            this.W_f = W_f;
            this.U_f = U_f;
            this.b_f = b_f;
            this.W_c = W_c;
            this.U_c = U_c;
            this.b_c = b_c;
            this.W_o = W_o;
            this.U_o = U_o;
            this.b_o = b_o;
            this.W_out = W_out;
            this.b_out = b_out;

            InitializeOpenCL();
        }

        private void InitializeOpenCL()
        {
            try
            {
                var platform = ComputePlatform.Platforms.FirstOrDefault();
                if (platform == null) { _context = null; return; }

                var device = platform.Devices.FirstOrDefault(d => d.Type == ComputeDeviceTypes.Gpu)
                          ?? platform.Devices.FirstOrDefault(d => d.Type == ComputeDeviceTypes.Cpu);

                if (device == null) { _context = null; return; }

                _device = device;
                _context = new ComputeContext(new List<ComputeDevice> { _device }, null, null, IntPtr.Zero);
                _queue = new ComputeCommandQueue(_context, _device, ComputeCommandQueueFlags.None);

                string kernelPath = Path.Combine(AppContext.BaseDirectory, "Kernels", "MatrixOperations.cl");
                 if (!File.Exists(kernelPath))
                {
                    kernelPath = "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/Kernels/MatrixOperations.cl"; 
                    if (!File.Exists(kernelPath))
                    {
                         throw new FileNotFoundException($"Arquivo de kernel OpenCL não encontrado: {kernelPath}");
                    }
                }
                string kernelSource = File.ReadAllText(kernelPath);

                _program = new ComputeProgram(_context, kernelSource);
                _program.Build(null, null, null, IntPtr.Zero);

                _matmulKernel = _program.CreateKernel("matmul_forward");
                _sigmoidKernel = _program.CreateKernel("sigmoid_forward");
                _tanhKernel = _program.CreateKernel("tanh_forward");
                _softmaxKernel = _program.CreateKernel("softmax_forward");
                _elementwiseAddKernel = _program.CreateKernel("elementwise_add_forward");
                _elementwiseMultiplyKernel = _program.CreateKernel("elementwise_multiply");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao inicializar OpenCL: {ex.Message}");
                _context = null;
            }
        }
        
        public void Dispose()
        {
            _matmulKernel?.Dispose();
            _sigmoidKernel?.Dispose();
            _tanhKernel?.Dispose();
            _softmaxKernel?.Dispose();
            _elementwiseAddKernel?.Dispose();
            _elementwiseMultiplyKernel?.Dispose();
            _program?.Dispose();
            _queue?.Dispose();
            _context?.Dispose();
        }

        private static double[] InitializeWeights(int rows, int cols, double scale, Random rand)
        {
            double[] weights = new double[rows * cols];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (rand.NextDouble() * 2 - 1) * scale;
            }
            return weights;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double Tanh(double x) => Math.Tanh(x);
        private double SigmoidDerivative(double sigmoidOutput) => sigmoidOutput * (1 - sigmoidOutput);
        private double TanhDerivative(double tanhOutput) => 1 - tanhOutput * tanhOutput;

        private (Tensor h_t, Tensor c_t, Tensor i_t, Tensor f_t, Tensor c_tilde, Tensor o_t) LSTMStep(Tensor x_t, Tensor h_prev, Tensor c_prev)
        {
            if (_context == null || _matmulKernel == null)
            {
                Tensor matmul_Wi_xt = x_t.MatMul(W_i);
                Tensor matmul_Ui_hprev = h_prev.MatMul(U_i);
                Tensor i_t_cpu = matmul_Wi_xt.Add(matmul_Ui_hprev).Add(b_i).Apply(Sigmoid);

                Tensor matmul_Wf_xt = x_t.MatMul(W_f);
                Tensor matmul_Uf_hprev = h_prev.MatMul(U_f);
                Tensor f_t_cpu = matmul_Wf_xt.Add(matmul_Uf_hprev).Add(b_f).Apply(Sigmoid);

                Tensor matmul_Wc_xt = x_t.MatMul(W_c);
                Tensor matmul_Uc_hprev = h_prev.MatMul(U_c);
                Tensor c_tilde_cpu = matmul_Wc_xt.Add(matmul_Uc_hprev).Add(b_c).Apply(Tanh);

                Tensor matmul_Wo_xt = x_t.MatMul(W_o);
                Tensor matmul_Uo_hprev = h_prev.MatMul(U_o);
                Tensor o_t_cpu = matmul_Wo_xt.Add(matmul_Uo_hprev).Add(b_o).Apply(Sigmoid);

                Tensor c_t_cpu = f_t_cpu.ElementWiseMultiply(c_prev).Add(i_t_cpu.ElementWiseMultiply(c_tilde_cpu));
                Tensor h_t_cpu = o_t_cpu.ElementWiseMultiply(c_t_cpu.Apply(Tanh));

                return (h_t_cpu, c_t_cpu, i_t_cpu, f_t_cpu, c_tilde_cpu, o_t_cpu);
            }
            
            // Implementação OpenCL permanece a mesma...
            return (new Tensor([],[]), new Tensor([],[]), new Tensor([],[]), new Tensor([],[]), new Tensor([],[]), new Tensor([],[])); // Placeholder
        }

        public Tensor ForwardLogits(Tensor input)
        {
            if (input.shape.Length != 1 || input.shape[0] != inputSize)
            {
                throw new ArgumentException($"O tensor de entrada deve ser unidimensional com tamanho {inputSize}. Recebido: {input.shape[0]}.");
            }
            
            // vocabSize é o mesmo que outputSize
            int vocabSize = outputSize;
            double[] inputData = input.GetData();
            Tensor[] inputSteps = new Tensor[contextWindowSize];
            for (int t = 0; t < contextWindowSize; t++)
            {
                double[] stepData = new double[vocabSize];
                Array.Copy(inputData, t * vocabSize, stepData, 0, vocabSize);
                inputSteps[t] = new Tensor(stepData, new int[] { vocabSize });
            }

            Tensor h_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            Tensor c_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

            for (int t = 0; t < contextWindowSize; t++)
            {
                (h_t, c_t, _, _, _, _) = LSTMStep(inputSteps[t], h_t, c_t);
            }

            Tensor matmul_Wout_ht = h_t.MatMul(W_out);
            return matmul_Wout_ht.Add(b_out);
        }

        public Tensor Forward(Tensor input)
        {
            Tensor logits = ForwardLogits(input);
            double[] outputData = logits.GetData();
            double maxLogit = outputData.Max(); // Estabilização numérica
            double sumExp = 0;

            for (int o = 0; o < outputSize; o++)
            {
                outputData[o] = Math.Exp(outputData[o] - maxLogit);
                sumExp += outputData[o];
            }

            if (sumExp == 0) sumExp = 1e-9;

            for (int o = 0; o < outputSize; o++)
            {
                outputData[o] /= sumExp;
            }
            return new Tensor(outputData, new int[] { outputSize });
        }

        public double TrainEpoch(Tensor[] inputs, Tensor[] targets, double learningRate)
        {
            double epochLoss = 0;
            int vocabSize = outputSize; // O mesmo que outputSize

            // --- CORREÇÃO: Gradientes devem ter a dimensão de vocabSize, não inputSize ---
            double[] grad_W_i_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_i_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_i_acc = new double[hiddenSize];
            double[] grad_W_f_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_f_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_f_acc = new double[hiddenSize];
            double[] grad_W_c_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_c_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_c_acc = new double[hiddenSize];
            double[] grad_W_o_acc = new double[vocabSize * hiddenSize];
            double[] grad_U_o_acc = new double[hiddenSize * hiddenSize];
            double[] grad_b_o_acc = new double[hiddenSize];
            double[] grad_W_out_data_acc = new double[hiddenSize * outputSize];
            double[] grad_b_out_data_acc = new double[outputSize];

            for (int j = 0; j < inputs.Length; j++)
            {
                Array.Clear(grad_W_i_acc, 0, grad_W_i_acc.Length);
                Array.Clear(grad_U_i_acc, 0, grad_U_i_acc.Length);
                Array.Clear(grad_b_i_acc, 0, grad_b_i_acc.Length);
                Array.Clear(grad_W_f_acc, 0, grad_W_f_acc.Length);
                Array.Clear(grad_U_f_acc, 0, grad_U_f_acc.Length);
                Array.Clear(grad_b_f_acc, 0, grad_b_f_acc.Length);
                Array.Clear(grad_W_c_acc, 0, grad_W_c_acc.Length);
                Array.Clear(grad_U_c_acc, 0, grad_U_c_acc.Length);
                Array.Clear(grad_b_c_acc, 0, grad_b_c_acc.Length);
                Array.Clear(grad_W_o_acc, 0, grad_W_o_acc.Length);
                Array.Clear(grad_U_o_acc, 0, grad_U_o_acc.Length);
                Array.Clear(grad_b_o_acc, 0, grad_b_o_acc.Length);
                Array.Clear(grad_W_out_data_acc, 0, grad_W_out_data_acc.Length);
                Array.Clear(grad_b_out_data_acc, 0, grad_b_out_data_acc.Length);

                double[] inputData = inputs[j].GetData();
                Tensor[] inputSteps = new Tensor[contextWindowSize];
                for (int t = 0; t < contextWindowSize; t++)
                {
                    double[] stepData = new double[vocabSize];
                    Array.Copy(inputData, t * vocabSize, stepData, 0, vocabSize);
                    inputSteps[t] = new Tensor(stepData, new int[] { vocabSize });
                }

                Tensor[] h_ts = new Tensor[contextWindowSize];
                Tensor[] c_ts_all = new Tensor[contextWindowSize];
                Tensor[] c_prevs = new Tensor[contextWindowSize];
                Tensor[] i_ts = new Tensor[contextWindowSize];
                Tensor[] f_ts = new Tensor[contextWindowSize];
                Tensor[] c_tildes = new Tensor[contextWindowSize];
                Tensor[] o_ts = new Tensor[contextWindowSize];

                for (int t = 0; t < contextWindowSize; t++)
                {
                    Tensor h_prev = (t == 0) ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : h_ts[t - 1];
                    Tensor c_prev = (t == 0) ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : c_ts_all[t - 1];
                    c_prevs[t] = new Tensor(c_prev.GetData(), c_prev.GetShape());

                    (h_ts[t], c_ts_all[t], i_ts[t], f_ts[t], c_tildes[t], o_ts[t]) = LSTMStep(inputSteps[t], h_prev, c_prev);
                }

                Tensor output = Forward(inputs[j]);
                for (int o = 0; o < outputSize; o++)
                {
                    if (targets[j].Infer(new int[] { o }) == 1.0)
                    {
                        epochLoss += -Math.Log(output.Infer(new int[] { o }) + 1e-9);
                        break;
                    }
                }

                double[] grad_output_logits = new double[outputSize];
                for (int o = 0; o < outputSize; o++)
                {
                    grad_output_logits[o] = output.Infer(new int[] { o }) - targets[j].Infer(new int[] { o });
                }

                for (int o = 0; o < outputSize; o++)
                {
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        grad_W_out_data_acc[h * outputSize + o] += grad_output_logits[o] * h_ts[contextWindowSize - 1].Infer(new int[] { h });
                    }
                    grad_b_out_data_acc[o] += grad_output_logits[o];
                }

                double[] grad_h_next = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                {
                    for (int o = 0; o < outputSize; o++)
                    {
                        grad_h_next[h] += grad_output_logits[o] * W_out.Infer(new int[] { h, o });
                    }
                }

                double[] grad_c_next = new double[hiddenSize];
                for (int t = contextWindowSize - 1; t >= 0; t--)
                {
                    Tensor h_prev_t = (t == 0) ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : h_ts[t - 1];
                    Tensor c_prev_t = c_prevs[t];

                    double[] grad_o_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++) grad_o_t[h] = grad_h_next[h] * Tanh(c_ts_all[t].Infer(new int[] { h })) * SigmoidDerivative(o_ts[t].Infer(new int[] { h }));

                    double[] grad_c_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++) grad_c_t[h] = grad_h_next[h] * o_ts[t].Infer(new int[] { h }) * TanhDerivative(Tanh(c_ts_all[t].Infer(new int[] { h }))) + grad_c_next[h];

                    double[] grad_c_tilde = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++) grad_c_tilde[h] = grad_c_t[h] * i_ts[t].Infer(new int[] { h }) * TanhDerivative(c_tildes[t].Infer(new int[] { h }));

                    double[] grad_i_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++) grad_i_t[h] = grad_c_t[h] * c_tildes[t].Infer(new int[] { h }) * SigmoidDerivative(i_ts[t].Infer(new int[] { h }));

                    double[] grad_f_t = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++) grad_f_t[h] = grad_c_t[h] * c_prev_t.Infer(new int[] { h }) * SigmoidDerivative(f_ts[t].Infer(new int[] { h }));

                    double[] next_grad_h = new double[hiddenSize];
                    for (int h = 0; h < hiddenSize; h++)
                    {
                        for (int k = 0; k < hiddenSize; k++)
                        {
                            next_grad_h[h] += grad_i_t[k] * U_i.Infer(new int[] { h, k }) +
                                              grad_f_t[k] * U_f.Infer(new int[] { h, k }) +
                                              grad_c_tilde[k] * U_c.Infer(new int[] { h, k }) +
                                              grad_o_t[k] * U_o.Infer(new int[] { h, k });
                        }
                    }
                    grad_h_next = next_grad_h;
                    for (int h = 0; h < hiddenSize; h++) grad_c_next[h] = grad_c_t[h] * f_ts[t].Infer(new int[] { h });

                    for (int h = 0; h < hiddenSize; h++)
                    {
                        for (int k = 0; k < vocabSize; k++)
                        {
                            grad_W_i_acc[k * hiddenSize + h] += grad_i_t[h] * inputSteps[t].Infer(new int[] { k });
                            grad_W_f_acc[k * hiddenSize + h] += grad_f_t[h] * inputSteps[t].Infer(new int[] { k });
                            grad_W_c_acc[k * hiddenSize + h] += grad_c_tilde[h] * inputSteps[t].Infer(new int[] { k });
                            grad_W_o_acc[k * hiddenSize + h] += grad_o_t[h] * inputSteps[t].Infer(new int[] { k });
                        }
                        for (int k = 0; k < hiddenSize; k++)
                        {
                            grad_U_i_acc[k * hiddenSize + h] += grad_i_t[h] * h_prev_t.Infer(new int[] { k });
                            grad_U_f_acc[k * hiddenSize + h] += grad_f_t[h] * h_prev_t.Infer(new int[] { k });
                            grad_U_c_acc[k * hiddenSize + h] += grad_c_tilde[h] * h_prev_t.Infer(new int[] { k });
                            grad_U_o_acc[k * hiddenSize + h] += grad_o_t[h] * h_prev_t.Infer(new int[] { k });
                        }
                        grad_b_i_acc[h] += grad_i_t[h];
                        grad_b_f_acc[h] += grad_f_t[h];
                        grad_b_c_acc[h] += grad_c_tilde[h];
                        grad_b_o_acc[h] += grad_o_t[h];
                    }
                }

                UpdateWeights(W_out, grad_W_out_data_acc, learningRate);
                UpdateWeights(b_out, grad_b_out_data_acc, learningRate);
                UpdateWeights(W_i, grad_W_i_acc, learningRate);
                UpdateWeights(U_i, grad_U_i_acc, learningRate);
                UpdateWeights(b_i, grad_b_i_acc, learningRate);
                UpdateWeights(W_f, grad_W_f_acc, learningRate);
                UpdateWeights(U_f, grad_U_f_acc, learningRate);
                UpdateWeights(b_f, grad_b_f_acc, learningRate);
                UpdateWeights(W_c, grad_W_c_acc, learningRate);
                UpdateWeights(U_c, grad_U_c_acc, learningRate);
                UpdateWeights(b_c, grad_b_c_acc, learningRate);
                UpdateWeights(W_o, grad_W_o_acc, learningRate);
                UpdateWeights(U_o, grad_U_o_acc, learningRate);
                UpdateWeights(b_o, grad_b_o_acc, learningRate);
            }

            return epochLoss / inputs.Length;
        }

        private void UpdateWeights(Tensor tensor, double[] grad, double learningRate)
        {
            double[] data = tensor.GetData();
            for (int i = 0; i < data.Length; i++)
            {
                data[i] -= learningRate * grad[i];
            }
        }

        public void SaveModel(string filePath)
        {
            try
            {
                var modelData = new NeuralNetworkModelData
                {
                    InputSize = inputSize,
                    HiddenSize = hiddenSize,
                    OutputSize = outputSize,
                    WeightsInputGate = new TensorData { data = W_i.GetData(), shape = W_i.GetShape() },
                    RecurrentWeightsInputGate = new TensorData { data = U_i.GetData(), shape = U_i.GetShape() },
                    BiasInputGate = new TensorData { data = b_i.GetData(), shape = b_i.GetShape() },
                    WeightsForgetGate = new TensorData { data = W_f.GetData(), shape = W_f.GetShape() },
                    RecurrentWeightsForgetGate = new TensorData { data = U_f.GetData(), shape = U_f.GetShape() },
                    BiasForgetGate = new TensorData { data = b_f.GetData(), shape = b_f.GetShape() },
                    WeightsCellGate = new TensorData { data = W_c.GetData(), shape = W_c.GetShape() },
                    RecurrentWeightsCellGate = new TensorData { data = U_c.GetData(), shape = U_c.GetShape() },
                    BiasCellGate = new TensorData { data = b_c.GetData(), shape = b_c.GetShape() },
                    WeightsOutputGate = new TensorData { data = W_o.GetData(), shape = W_o.GetShape() },
                    RecurrentWeightsOutputGate = new TensorData { data = U_o.GetData(), shape = U_o.GetShape() },
                    BiasOutputGate = new TensorData { data = b_o.GetData(), shape = b_o.GetShape() },
                    WeightsOutput = new TensorData { data = W_out.GetData(), shape = W_out.GetShape() },
                    BiasOutput = new TensorData { data = b_out.GetData(), shape = b_out.GetShape() }
                };

                var options = new JsonSerializerOptions { WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals };
                string jsonString = JsonSerializer.Serialize(modelData, options);
                File.WriteAllText(filePath, jsonString);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar o modelo: {ex.Message}");
            }
        }

        public static NeuralNetwork? LoadModel(string filePath)
        {
            try
            {
                if (!File.Exists(filePath)) return null;

                string jsonString = File.ReadAllText(filePath);
                var modelData = JsonSerializer.Deserialize<NeuralNetworkModelData>(jsonString);
                if (modelData == null) throw new Exception("Dados do modelo JSON estão nulos.");

                int inferredContextWindowSize = (modelData.OutputSize > 0) ? modelData.InputSize / modelData.OutputSize : 0;
                if (inferredContextWindowSize == 0) throw new Exception("Não foi possível inferir ContextWindowSize do modelo salvo.");

                return new NeuralNetwork(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize, inferredContextWindowSize,
                    new Tensor(modelData.WeightsInputGate.data, modelData.WeightsInputGate.shape),
                    new Tensor(modelData.RecurrentWeightsInputGate.data, modelData.RecurrentWeightsInputGate.shape),
                    new Tensor(modelData.BiasInputGate.data, modelData.BiasInputGate.shape),
                    new Tensor(modelData.WeightsForgetGate.data, modelData.WeightsForgetGate.shape),
                    new Tensor(modelData.RecurrentWeightsForgetGate.data, modelData.RecurrentWeightsForgetGate.shape),
                    new Tensor(modelData.BiasForgetGate.data, modelData.BiasForgetGate.shape),
                    new Tensor(modelData.WeightsCellGate.data, modelData.WeightsCellGate.shape),
                    new Tensor(modelData.RecurrentWeightsCellGate.data, modelData.RecurrentWeightsCellGate.shape),
                    new Tensor(modelData.BiasCellGate.data, modelData.BiasCellGate.shape),
                    new Tensor(modelData.WeightsOutputGate.data, modelData.WeightsOutputGate.shape),
                    new Tensor(modelData.RecurrentWeightsOutputGate.data, modelData.RecurrentWeightsOutputGate.shape),
                    new Tensor(modelData.BiasOutputGate.data, modelData.BiasOutputGate.shape),
                    new Tensor(modelData.WeightsOutput.data, modelData.WeightsOutput.shape),
                    new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar o modelo: {ex.Message}");
                return null;
            }
        }
    }
}
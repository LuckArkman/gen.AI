using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Core;

public class NeuralNetwork
{
    private Tensor W_i; // Pesos da porta de entrada (input gate)
    private Tensor U_i; // Pesos recorrentes da porta de entrada
    private Tensor b_i; // Bias da porta de entrada
    private Tensor W_f; // Pesos da porta de esquecimento (forget gate)
    private Tensor U_f; // Pesos recorrentes da porta de esquecimento
    private Tensor b_f; // Bias da porta de esquecimento
    private Tensor W_c; // Pesos da porta da célula (cell gate)
    private Tensor U_c; // Pesos recorrentes da porta da célula
    private Tensor b_c; // Bias da porta da célula
    private Tensor W_o; // Pesos da porta de saída (output gate)
    private Tensor U_o; // Pesos recorrentes da porta de saída
    private Tensor b_o; // Bias da porta de saída
    private Tensor W_out; // Pesos da camada de saída
    private Tensor b_out; // Bias da camada de saída
    private readonly int inputSize; // vocabSize * contextWindowSize
    private readonly int hiddenSize;
    private readonly int outputSize; // vocabSize
    private readonly int contextWindowSize; // Tamanho da janela de contexto

    public int InputSize => inputSize;
    public int HiddenSize => hiddenSize;
    public int OutputSize => outputSize;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int contextWindowSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.contextWindowSize = contextWindowSize;

        Random rand = new Random();
        double sqrtFanIn = Math.Sqrt(2.0 / (inputSize + hiddenSize)); // Inicialização He ajustada

        // Inicialização dos pesos e biases para as portas da LSTM
        W_i = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
        U_i = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
        b_i = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
        W_f = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
        U_f = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
        b_f = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
        W_c = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
        U_c = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
        b_c = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
        W_o = new Tensor(InitializeWeights(inputSize, hiddenSize, sqrtFanIn, rand), new int[] { inputSize, hiddenSize });
        U_o = new Tensor(InitializeWeights(hiddenSize, hiddenSize, sqrtFanIn, rand), new int[] { hiddenSize, hiddenSize });
        b_o = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
        W_out = new Tensor(InitializeWeights(hiddenSize, outputSize, Math.Sqrt(2.0 / hiddenSize), rand), new int[] { hiddenSize, outputSize });
        b_out = new Tensor(new double[outputSize], new int[] { outputSize });
    }

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

    public Tensor ForwardLogits(Tensor input)
    {
        if (input.shape.Length != 1 || input.shape[0] != inputSize)
        {
            throw new ArgumentException($"O tensor de entrada deve ser unidimensional com tamanho {inputSize}. Recebido: {input.shape[0]}.");
        }

        // Divide a entrada em contextWindowSize vetores de tamanho vocabSize
        int vocabSize = inputSize / contextWindowSize;
        double[] inputData = input.GetData();
        Tensor[] inputSteps = new Tensor[contextWindowSize];
        for (int t = 0; t < contextWindowSize; t++)
        {
            double[] stepData = new double[vocabSize];
            Array.Copy(inputData, t * vocabSize, stepData, 0, vocabSize);
            inputSteps[t] = new Tensor(stepData, new int[] { vocabSize });
        }

        // Inicializa estados da LSTM
        Tensor h_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
        Tensor c_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });

        // Processa cada passo de tempo
        for (int t = 0; t < contextWindowSize; t++)
        {
            (h_t, c_t, _, _, _, _) = LSTMStep(inputSteps[t], h_t, c_t);
        }

        // Camada de saída
        double[] outputData = new double[outputSize];
        double[] h_t_data = h_t.GetData(); // Cache para desempenho
        for (int o = 0; o < outputSize; o++)
        {
            double sum = 0;
            for (int h = 0; h < hiddenSize; h++)
            {
                sum += h_t_data[h] * W_out.Infer(new int[] { h, o });
            }
            sum += b_out.Infer(new int[] { o });
            outputData[o] = sum; // Logits
        }

        return new Tensor(outputData, new int[] { outputSize });
    }

    private (Tensor h_t, Tensor c_t, Tensor i_t, Tensor f_t, Tensor c_tilde, Tensor o_t) LSTMStep(Tensor x_t, Tensor h_prev, Tensor c_prev)
    {
        double[] x_t_data = x_t.GetData();
        double[] h_prev_data = h_prev.GetData();
        double[] c_prev_data = c_prev.GetData();

        // Porta de entrada
        double[] i_t_data = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < x_t.shape[0]; i++)
            {
                sum += x_t_data[i] * W_i.Infer(new int[] { i, h });
            }
            for (int h_prev_idx = 0; h_prev_idx < hiddenSize; h_prev_idx++)
            {
                sum += h_prev_data[h_prev_idx] * U_i.Infer(new int[] { h_prev_idx, h });
            }
            sum += b_i.Infer(new int[] { h });
            i_t_data[h] = Sigmoid(sum);
        }
        Tensor i_t = new Tensor(i_t_data, new int[] { hiddenSize });

        // Porta de esquecimento
        double[] f_t_data = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < x_t.shape[0]; i++)
            {
                sum += x_t_data[i] * W_f.Infer(new int[] { i, h });
            }
            for (int h_prev_idx = 0; h_prev_idx < hiddenSize; h_prev_idx++)
            {
                sum += h_prev_data[h_prev_idx] * U_f.Infer(new int[] { h_prev_idx, h });
            }
            sum += b_f.Infer(new int[] { h });
            f_t_data[h] = Sigmoid(sum);
        }
        Tensor f_t = new Tensor(f_t_data, new int[] { hiddenSize });

        // Porta da célula (candidato)
        double[] c_tilde_data = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < x_t.shape[0]; i++)
            {
                sum += x_t_data[i] * W_c.Infer(new int[] { i, h });
            }
            for (int h_prev_idx = 0; h_prev_idx < hiddenSize; h_prev_idx++)
            {
                sum += h_prev_data[h_prev_idx] * U_c.Infer(new int[] { h_prev_idx, h });
            }
            sum += b_c.Infer(new int[] { h });
            c_tilde_data[h] = Tanh(sum);
        }
        Tensor c_tilde = new Tensor(c_tilde_data, new int[] { hiddenSize });

        // Estado da célula
        double[] c_t_data = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            c_t_data[h] = f_t_data[h] * c_prev_data[h] + i_t_data[h] * c_tilde_data[h];
        }
        Tensor c_t = new Tensor(c_t_data, new int[] { hiddenSize });

        // Porta de saída
        double[] o_t_data = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < x_t.shape[0]; i++)
            {
                sum += x_t_data[i] * W_o.Infer(new int[] { i, h });
            }
            for (int h_prev_idx = 0; h_prev_idx < hiddenSize; h_prev_idx++)
            {
                sum += h_prev_data[h_prev_idx] * U_o.Infer(new int[] { h_prev_idx, h });
            }
            sum += b_o.Infer(new int[] { h });
            o_t_data[h] = Sigmoid(sum);
        }
        Tensor o_t = new Tensor(o_t_data, new int[] { hiddenSize });

        // Estado oculto
        double[] h_t_data = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++)
        {
            h_t_data[h] = o_t_data[h] * Tanh(c_t_data[h]);
        }
        Tensor h_t = new Tensor(h_t_data, new int[] { hiddenSize });

        return (h_t, c_t, i_t, f_t, c_tilde, o_t);
    }

    public Tensor Forward(Tensor input)
    {
        Tensor logits = ForwardLogits(input);
        double[] outputData = logits.GetData();

        double sumExp = 0;
        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] = Math.Exp(outputData[o]);
            sumExp += outputData[o];
        }

        if (sumExp == 0) sumExp = 1e-9;
        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] /= sumExp; // Softmax
        }

        return new Tensor(outputData, new int[] { outputSize });
    }

    public double TrainEpoch(Tensor[] inputs, Tensor[] targets, double learningRate)
    {
        double epochLoss = 0;

        for (int j = 0; j < inputs.Length; j++)
        {
            int vocabSize = inputSize / contextWindowSize;
            double[] inputData = inputs[j].GetData();
            Tensor[] inputSteps = new Tensor[contextWindowSize];
            for (int t = 0; t < contextWindowSize; t++)
            {
                double[] stepData = new double[vocabSize];
                Array.Copy(inputData, t * vocabSize, stepData, 0, vocabSize);
                inputSteps[t] = new Tensor(stepData, new int[] { vocabSize });
            }

            // Forward pass
            Tensor h_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            Tensor c_t = new Tensor(new double[hiddenSize], new int[] { hiddenSize });
            Tensor[] i_ts = new Tensor[contextWindowSize];
            Tensor[] f_ts = new Tensor[contextWindowSize];
            Tensor[] c_tildes = new Tensor[contextWindowSize];
            Tensor[] c_ts = new Tensor[contextWindowSize];
            Tensor[] o_ts = new Tensor[contextWindowSize];
            Tensor[] h_ts = new Tensor[contextWindowSize];
            Tensor[] c_prevs = new Tensor[contextWindowSize];

            for (int t = 0; t < contextWindowSize; t++)
            {
                Tensor h_prev = t == 0 ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : h_ts[t - 1];
                Tensor c_prev = t == 0 ? new Tensor(new double[hiddenSize], new int[] { hiddenSize }) : c_ts[t - 1];
                (h_t, c_t, i_ts[t], f_ts[t], c_tildes[t], o_ts[t]) = LSTMStep(inputSteps[t], h_prev, c_prev);
                h_ts[t] = new Tensor(h_t.GetData(), new int[] { hiddenSize });
                c_ts[t] = new Tensor(c_t.GetData(), new int[] { hiddenSize });
                c_prevs[t] = new Tensor(c_prev.GetData(), new int[] { hiddenSize });
            }

            // Camada de saída
            Tensor output = Forward(inputs[j]);
            for (int o = 0; o < outputSize; o++)
            {
                if (targets[j].Infer(new int[] { o }) == 1.0)
                {
                    double outputValue = output.Infer(new int[] { o });
                    epochLoss += -Math.Log(outputValue + 1e-9);
                    break;
                }
            }

            // Backward pass (BPTT)
            double[] grad_output = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                grad_output[o] = output.Infer(new int[] { o }) - targets[j].Infer(new int[] { o });
            }

            double[] grad_W_out = new double[hiddenSize * outputSize];
            double[] grad_b_out = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                for (int h = 0; h < hiddenSize; h++)
                {
                    grad_W_out[h * outputSize + o] = grad_output[o] * h_ts[contextWindowSize - 1].Infer(new int[] { h });
                }
                grad_b_out[o] = grad_output[o];
            }

            double[] grad_h_next = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++)
            {
                double sum = 0;
                for (int o = 0; o < outputSize; o++)
                {
                    sum += grad_output[o] * W_out.Infer(new int[] { h, o });
                }
                grad_h_next[h] = sum;
            }

            double[] grad_c_next = new double[hiddenSize];
            double[][] grad_i_ts = new double[contextWindowSize][];
            double[][] grad_f_ts = new double[contextWindowSize][];
            double[][] grad_c_tildes = new double[contextWindowSize][];
            double[][] grad_o_ts = new double[contextWindowSize][];

            for (int t = contextWindowSize - 1; t >= 0; t--)
            {
                grad_i_ts[t] = new double[hiddenSize];
                grad_f_ts[t] = new double[hiddenSize];
                grad_c_tildes[t] = new double[hiddenSize];
                grad_o_ts[t] = new double[hiddenSize];

                double[] grad_h_t = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                {
                    grad_h_t[h] = t == contextWindowSize - 1 ? grad_h_next[h] : 0;
                }

                double[] grad_c_t = new double[hiddenSize];
                for (int h = 0; h < hiddenSize; h++)
                {
                    grad_c_t[h] = grad_h_t[h] * o_ts[t].Infer(new int[] { h }) * TanhDerivative(Tanh(c_ts[t].Infer(new int[] { h }))) +
                                  (t < contextWindowSize - 1 ? grad_c_next[h] * f_ts[t + 1].Infer(new int[] { h }) : 0);
                }

                for (int h = 0; h < hiddenSize; h++)
                {
                    grad_o_ts[t][h] = grad_h_t[h] * Tanh(c_ts[t].Infer(new int[] { h })) * SigmoidDerivative(o_ts[t].Infer(new int[] { h }));
                    grad_c_tildes[t][h] = grad_c_t[h] * i_ts[t].Infer(new int[] { h }) * TanhDerivative(c_tildes[t].Infer(new int[] { h }));
                    grad_i_ts[t][h] = grad_c_t[h] * c_tildes[t].Infer(new int[] { h }) * SigmoidDerivative(i_ts[t].Infer(new int[] { h }));
                    grad_f_ts[t][h] = grad_c_t[h] * c_prevs[t].Infer(new int[] { h }) * SigmoidDerivative(f_ts[t].Infer(new int[] { h }));
                }

                grad_c_next = grad_c_t;
            }

            double[] grad_W_i = new double[inputSize * hiddenSize];
            double[] grad_U_i = new double[hiddenSize * hiddenSize];
            double[] grad_b_i = new double[hiddenSize];
            double[] grad_W_f = new double[inputSize * hiddenSize];
            double[] grad_U_f = new double[hiddenSize * hiddenSize];
            double[] grad_b_f = new double[hiddenSize];
            double[] grad_W_c = new double[inputSize * hiddenSize];
            double[] grad_U_c = new double[hiddenSize * hiddenSize];
            double[] grad_b_c = new double[hiddenSize];
            double[] grad_W_o = new double[inputSize * hiddenSize];
            double[] grad_U_o = new double[hiddenSize * hiddenSize];
            double[] grad_b_o = new double[hiddenSize];

            for (int t = 0; t < contextWindowSize; t++)
            {
                double[] input_t_data = inputSteps[t].GetData();
                double[] h_prev_data = t == 0 ? new double[hiddenSize] : h_ts[t - 1].GetData();
                for (int h = 0; h < hiddenSize; h++)
                {
                    for (int i = 0; i < inputSteps[t].shape[0]; i++)
                    {
                        grad_W_i[i * hiddenSize + h] += grad_i_ts[t][h] * input_t_data[j];
                        grad_W_f[i * hiddenSize + h] += grad_f_ts[t][h] * input_t_data[j];
                        grad_W_c[i * hiddenSize + h] += grad_c_tildes[t][h] * input_t_data[j];
                        grad_W_o[i * hiddenSize + h] += grad_o_ts[t][h] * input_t_data[j];
                    }
                    for (int h_prev = 0; h_prev < hiddenSize; h_prev++)
                    {
                        grad_U_i[h_prev * hiddenSize + h] += grad_i_ts[t][h] * h_prev_data[h_prev];
                        grad_U_f[h_prev * hiddenSize + h] += grad_f_ts[t][h] * h_prev_data[h_prev];
                        grad_U_c[h_prev * hiddenSize + h] += grad_c_tildes[t][h] * h_prev_data[h_prev];
                        grad_U_o[h_prev * hiddenSize + h] += grad_o_ts[t][h] * h_prev_data[h_prev];
                    }
                    grad_b_i[h] += grad_i_ts[t][h];
                    grad_b_f[h] += grad_f_ts[t][h];
                    grad_b_c[h] += grad_c_tildes[t][h];
                    grad_b_o[h] += grad_o_ts[t][h];
                }
            }

            // Atualiza pesos
            UpdateWeights(W_i, grad_W_i, learningRate);
            UpdateWeights(U_i, grad_U_i, learningRate);
            UpdateWeights(b_i, grad_b_i, learningRate);
            UpdateWeights(W_f, grad_W_f, learningRate);
            UpdateWeights(U_f, grad_U_f, learningRate);
            UpdateWeights(b_f, grad_b_f, learningRate);
            UpdateWeights(W_c, grad_W_c, learningRate);
            UpdateWeights(U_c, grad_U_c, learningRate);
            UpdateWeights(b_c, grad_b_c, learningRate);
            UpdateWeights(W_o, grad_W_o, learningRate);
            UpdateWeights(U_o, grad_U_o, learningRate);
            UpdateWeights(b_o, grad_b_o, learningRate);
            UpdateWeights(W_out, grad_W_out, learningRate);
            UpdateWeights(b_out, grad_b_out, learningRate);
        }

        return epochLoss / inputs.Length;
    }

    private void UpdateWeights(Tensor tensor, double[] grad, double learningRate)
    {
        double[] data = tensor.GetData();
        double[] updatedData = new double[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            updatedData[i] = data[i] - learningRate * grad[i];
        }
        tensor.SetData(updatedData);
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

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
            };
            string jsonString = JsonSerializer.Serialize(modelData, options);
            File.WriteAllText(filePath, jsonString);
            Console.WriteLine($"Modelo salvo em JSON: {filePath}");
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
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Arquivo do modelo não encontrado: {filePath}");
                return null;
            }

            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelData>(jsonString);
            if (modelData == null || 
                modelData.WeightsInputGate?.data == null || modelData.RecurrentWeightsInputGate?.data == null || modelData.BiasInputGate?.data == null ||
                modelData.WeightsForgetGate?.data == null || modelData.RecurrentWeightsForgetGate?.data == null || modelData.BiasForgetGate?.data == null ||
                modelData.WeightsCellGate?.data == null || modelData.RecurrentWeightsCellGate?.data == null || modelData.BiasCellGate?.data == null ||
                modelData.WeightsOutputGate?.data == null || modelData.RecurrentWeightsOutputGate?.data == null || modelData.BiasOutputGate?.data == null ||
                modelData.WeightsOutput?.data == null || modelData.BiasOutput?.data == null)
            {
                throw new Exception("Dados do modelo JSON estão incompletos.");
            }

            Tensor? W_i = new Tensor(modelData.WeightsInputGate.data, modelData.WeightsInputGate.shape);
            Tensor? U_i = new Tensor(modelData.RecurrentWeightsInputGate.data, modelData.RecurrentWeightsInputGate.shape);
            Tensor? b_i = new Tensor(modelData.BiasInputGate.data, modelData.BiasInputGate.shape);
            Tensor? W_f = new Tensor(modelData.WeightsForgetGate.data, modelData.WeightsForgetGate.shape);
            Tensor? U_f = new Tensor(modelData.RecurrentWeightsForgetGate.data, modelData.RecurrentWeightsForgetGate.shape);
            Tensor? b_f = new Tensor(modelData.BiasForgetGate.data, modelData.BiasForgetGate.shape);
            Tensor? W_c = new Tensor(modelData.WeightsCellGate.data, modelData.WeightsCellGate.shape);
            Tensor? U_c = new Tensor(modelData.RecurrentWeightsCellGate.data, modelData.RecurrentWeightsCellGate.shape);
            Tensor? b_c = new Tensor(modelData.BiasCellGate.data, modelData.BiasCellGate.shape);
            Tensor? W_o = new Tensor(modelData.WeightsOutputGate.data, modelData.WeightsOutputGate.shape);
            Tensor? U_o = new Tensor(modelData.RecurrentWeightsOutputGate.data, modelData.RecurrentWeightsOutputGate.shape);
            Tensor? b_o = new Tensor(modelData.BiasOutputGate.data, modelData.BiasOutputGate.shape);
            Tensor? W_out = new Tensor(modelData.WeightsOutput.data, modelData.WeightsOutput.shape);
            Tensor? b_out = new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);

            return new NeuralNetwork(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize, modelData.InputSize / modelData.OutputSize,
                W_i, U_i, b_i, W_f, U_f, b_f, W_c, U_c, b_c, W_o, U_o, b_o, W_out, b_out);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o modelo: {ex.Message}");
            return null;
        }
    }
}
using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Core;

public class NeuralNetwork
{
    private Tensor weightsHidden;
    private Tensor biasHidden;
    private Tensor weightsOutput;
    private Tensor biasOutput;
    private readonly int inputSize;
    private readonly int intHiddenSize; // Renomeado para evitar conflito com propriedade
    private readonly int outputSize;

    public int InputSize => inputSize;
    public int HiddenSize => intHiddenSize; // Usa o novo nome aqui
    public int OutputSize => outputSize;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.intHiddenSize = hiddenSize; // Atribuído ao novo nome
        this.outputSize = outputSize;

        Random rand = new Random();
        double sqrtFanInHidden = Math.Sqrt(2.0 / inputSize); // He initialization for ReLU
        double sqrtFanInOutput = Math.Sqrt(2.0 / hiddenSize); // He initialization for ReLU (or Xavier for linear/softmax)

        double[] weightsHiddenData = new double[inputSize * intHiddenSize];
        double[] biasHiddenData = new double[intHiddenSize];
        double[] weightsOutputData = new double[intHiddenSize * outputSize];
        double[] biasOutputData = new double[outputSize];

        for (int i = 0; i < weightsHiddenData.Length; i++)
            weightsHiddenData[i] = (rand.NextDouble() * 2 - 1) * sqrtFanInHidden; // Scaled initialization
        for (int i = 0; i < biasHiddenData.Length; i++)
            biasHiddenData[i] = 0; // Common to initialize biases to zero or small positive
        for (int i = 0; i < weightsOutputData.Length; i++)
            weightsOutputData[i] = (rand.NextDouble() * 2 - 1) * sqrtFanInOutput; // Scaled initialization
        for (int i = 0; i < biasOutputData.Length; i++)
            biasOutputData[i] = 0; // Common to initialize biases to zero or small positive

        weightsHidden = new Tensor(weightsHiddenData, new int[] { inputSize, intHiddenSize });
        biasHidden = new Tensor(biasHiddenData, new int[] { intHiddenSize });
        weightsOutput = new Tensor(weightsOutputData, new int[] { intHiddenSize, outputSize });
        biasOutput = new Tensor(biasOutputData, new int[] { outputSize });
    }

    private NeuralNetwork(int inputSize, int hiddenSize, int outputSize,
        Tensor weightsHidden, Tensor biasHidden,
        Tensor weightsOutput, Tensor biasOutput)
    {
        this.inputSize = inputSize;
        this.intHiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.weightsHidden = weightsHidden;
        this.biasHidden = biasHidden;
        this.weightsOutput = weightsOutput;
        this.biasOutput = biasOutput;
    }

    public Tensor ForwardLogits(Tensor input)
    {
        if (input.shape.Length != 1 || input.shape[0] != inputSize)
        {
            throw new ArgumentException(
                $"O tensor de entrada deve ser unidimensional com tamanho igual a inputSize ({inputSize}). Mas recebeu {input.shape[0]}.");
        }

        // Camada Oculta (ReLU)
        double[] hiddenData = new double[intHiddenSize];
        for (int h = 0; h < intHiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < inputSize; i++)
            {
                sum += input.Infer(new int[] { i }) * weightsHidden.Infer(new int[] { i, h });
            }

            sum += biasHidden.Infer(new int[] { h });
            hiddenData[h] = Math.Max(0, sum); // ReLU
        }

        Tensor hidden = new Tensor(hiddenData, new int[] { intHiddenSize });

        // Camada de Saída (Linear para logits)
        double[] outputLogitsData = new double[outputSize];
        for (int o = 0; o < outputSize; o++)
        {
            double sum = 0;
            for (int h = 0; h < intHiddenSize; h++)
            {
                sum += hidden.Infer(new int[] { h }) * weightsOutput.Infer(new int[] { h, o });
            }
            sum += biasOutput.Infer(new int[] { o });
            outputLogitsData[o] = sum;
        }

        return new Tensor(outputLogitsData, new int[] { outputSize });
    }


    public Tensor Forward(Tensor input)
    {
        Tensor logits = ForwardLogits(input); // Obtém os logits primeiro
        double[] outputData = logits.GetData();

        double sumExp = 0;
        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] = Math.Exp(outputData[o]); // Aplica exponencial para Softmax
            sumExp += outputData[o];
        }

        // Evita divisão por zero se sumExp for muito pequeno
        if (sumExp == 0) sumExp = 1e-9; 

        for (int o = 0; o < outputSize; o++)
        {
            outputData[o] /= sumExp; // Normaliza para Softmax
        }

        return new Tensor(outputData, new int[] { outputSize });
    }

    public double TrainEpoch(Tensor[] inputs, Tensor[] targets, double learningRate)
    {
        double epochLoss = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            // --- 1. Forward Pass (Propagação Direta) ---
            // Calcular a saída da camada oculta (usando ComputeHidden para reutilização)
            Tensor hiddenInputBeforeActivation = ComputeHiddenInput(inputs[i]); // A entrada da ReLU
            Tensor hiddenActivation = ComputeHidden(inputs[i]); // A saída da ReLU

            // Calcular a saída da rede (probabilidades Softmax)
            Tensor output = Forward(inputs[i]); 

            // --- 2. Calcular a Perda para esta amostra ---
            for (int o = 0; o < outputSize; o++)
            {
                if (targets[i].Infer(new int[] { o }) == 1.0)
                {
                    double outputValue = output.Infer(new int[] { o });
                    if (outputValue <= 0)
                    {
                        epochLoss += -Math.Log(1e-9); 
                    }
                    else
                    {
                        epochLoss += -Math.Log(outputValue + 1e-9); // Perda de Entropia Cruzada
                    }
                    break;
                }
            }

            // --- 3. Backward Pass (Retropropagação) ---

            // a. Gradientes da Camada de Saída
            // dL/d(output_logits) = output_probabilities - target_one_hot
            double[] gradOutputLogits = new double[outputSize];
            for (int o = 0; o < outputSize; o++)
            {
                gradOutputLogits[o] = output.Infer(new int[] { o }) - targets[i].Infer(new int[] { o });
            }

            // b. Gradientes para os Pesos (weightsOutput) e Bias (biasOutput) da Camada de Saída
            // dW_output = dL/d(output_logits) * hidden_activation.T
            double[] gradWeightsOutputData = new double[intHiddenSize * outputSize];
            double[] gradBiasOutputData = new double[outputSize];

            for (int o = 0; o < outputSize; o++)
            {
                for (int h = 0; h < intHiddenSize; h++)
                {
                    int idx = h * outputSize + o;
                    gradWeightsOutputData[idx] = gradOutputLogits[o] * hiddenActivation.Infer(new int[] { h });
                }
                gradBiasOutputData[o] = gradOutputLogits[o];
            }

            // c. Propagar gradientes para a Camada Oculta (antes da ativação ReLU)
            // dL/d(hidden_input_before_activation) = dL/d(output_logits) * weightsOutput.T * ReLU_derivative(hidden_input_before_activation)
            double[] gradHiddenInput = new double[intHiddenSize];
            for (int h = 0; h < intHiddenSize; h++)
            {
                double sumErrorPropagated = 0;
                for (int o = 0; o < outputSize; o++)
                {
                    sumErrorPropagated += gradOutputLogits[o] * weightsOutput.Infer(new int[] { h, o });
                }
                // Aplica a derivada da ReLU: 1 se a entrada for > 0, 0 caso contrário.
                gradHiddenInput[h] = sumErrorPropagated * (hiddenInputBeforeActivation.Infer(new int[] { h }) > 0 ? 1 : 0);
            }

            // d. Gradientes para os Pesos (weightsHidden) e Bias (biasHidden) da Camada Oculta
            // dW_hidden = dL/d(hidden_input_before_activation) * input.T
            double[] gradWeightsHiddenData = new double[inputSize * intHiddenSize];
            double[] gradBiasHiddenData = new double[intHiddenSize];

            for (int h = 0; h < intHiddenSize; h++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    int idx = j * intHiddenSize + h;
                    gradWeightsHiddenData[idx] = gradHiddenInput[h] * inputs[i].Infer(new int[] { j });
                }
                gradBiasHiddenData[h] = gradHiddenInput[h];
            }

            // --- 4. Atualizar Pesos e Bias (Gradient Descent) ---
            // Aplica as atualizações APÓS todos os gradientes serem calculados para esta amostra
            
            // Atualizar pesos e bias da camada de saída
            double[] currentWeightsOutputData = weightsOutput.GetData();
            double[] currentBiasOutputData = biasOutput.GetData();
            double[] updatedWeightsOutputData = new double[currentWeightsOutputData.Length];
            double[] updatedBiasOutputData = new double[currentBiasOutputData.Length];

            for(int k = 0; k < currentWeightsOutputData.Length; k++)
            {
                updatedWeightsOutputData[k] = currentWeightsOutputData[k] - learningRate * gradWeightsOutputData[k];
            }
            for(int k = 0; k < currentBiasOutputData.Length; k++)
            {
                updatedBiasOutputData[k] = currentBiasOutputData[k] - learningRate * gradBiasOutputData[k];
            }

            weightsOutput = new Tensor(updatedWeightsOutputData, new int[] { intHiddenSize, outputSize });
            biasOutput = new Tensor(updatedBiasOutputData, new int[] { outputSize });

            // Atualizar pesos e bias da camada oculta
            double[] currentWeightsHiddenData = weightsHidden.GetData();
            double[] currentBiasHiddenData = biasHidden.GetData();
            double[] updatedWeightsHiddenData = new double[currentWeightsHiddenData.Length];
            double[] updatedBiasHiddenData = new double[currentBiasHiddenData.Length];
            
            for(int k = 0; k < currentWeightsHiddenData.Length; k++)
            {
                updatedWeightsHiddenData[k] = currentWeightsHiddenData[k] - learningRate * gradWeightsHiddenData[k];
            }
            for(int k = 0; k < currentBiasHiddenData.Length; k++)
            {
                updatedBiasHiddenData[k] = currentBiasHiddenData[k] - learningRate * gradBiasHiddenData[k];
            }

            weightsHidden = new Tensor(updatedWeightsHiddenData, new int[] { inputSize, intHiddenSize });
            biasHidden = new Tensor(updatedBiasHiddenData, new int[] { intHiddenSize });
        }

        return epochLoss / inputs.Length;
    }

    // Método auxiliar para calcular a saída da ReLU da camada oculta
    private Tensor ComputeHidden(Tensor input)
    {
        double[] hiddenData = new double[intHiddenSize];
        for (int h = 0; h < intHiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < inputSize; i++)
            {
                sum += input.Infer(new int[] { i }) * weightsHidden.Infer(new int[] { i, h });
            }

            sum += biasHidden.Infer(new int[] { h });
            hiddenData[h] = Math.Max(0, sum); // ReLU
        }

        return new Tensor(hiddenData, new int[] { intHiddenSize });
    }

    // NOVO Método auxiliar para calcular a entrada da ReLU da camada oculta (antes da ativação)
    private Tensor ComputeHiddenInput(Tensor input)
    {
        double[] hiddenInputData = new double[intHiddenSize];
        for (int h = 0; h < intHiddenSize; h++)
        {
            double sum = 0;
            for (int i = 0; i < inputSize; i++)
            {
                sum += input.Infer(new int[] { i }) * weightsHidden.Infer(new int[] { i, h });
            }

            sum += biasHidden.Infer(new int[] { h });
            hiddenInputData[h] = sum; // Sem ReLU aqui
        }

        return new Tensor(hiddenInputData, new int[] { intHiddenSize });
    }

    public void SaveModel(string filePath)
    {
        try
        {
            var modelData = new NeuralNetworkModelData
            {
                InputSize = inputSize,
                HiddenSize = intHiddenSize,
                OutputSize = outputSize,
                WeightsHidden = new TensorData { data = weightsHidden.GetData(), shape = weightsHidden.GetShape() },
                BiasHidden = new TensorData { data = biasHidden.GetData(), shape = biasHidden.GetShape() },
                WeightsOutput = new TensorData { data = weightsOutput.GetData(), shape = weightsOutput.GetShape() },
                BiasOutput = new TensorData { data = biasOutput.GetData(), shape = biasOutput.GetShape() }
            };

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
            };
            string jsonString = JsonSerializer.Serialize(modelData, options);

            File.WriteAllText(filePath, jsonString);
            Console.WriteLine($"Modelo salvo em JSON (System.Text.Json) em: {filePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao salvar o modelo em JSON (System.Text.Json): {ex.Message}");
        }
    }

    public static NeuralNetwork? LoadModel(string filePath)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Arquivo do modelo JSON não encontrado em: {filePath}");
                return null;
            }

            string jsonString = File.ReadAllText(filePath);
            var modelData = JsonSerializer.Deserialize<NeuralNetworkModelData>(jsonString);
            if (modelData == null)
            {
                throw new Exception("Falha ao desserializar dados do modelo JSON.");
            }

            if (modelData.WeightsHidden?.data == null || modelData.WeightsHidden.shape == null ||
                modelData.BiasHidden?.data == null || modelData.BiasHidden.shape == null ||
                modelData.WeightsOutput?.data == null || modelData.WeightsOutput.shape == null ||
                modelData.BiasOutput?.data == null || modelData.BiasOutput.shape == null)
            {
                throw new Exception("Dados do modelo JSON estão incompletos.");
            }

            Tensor loadedWeightsHidden = new Tensor(modelData.WeightsHidden.data, modelData.WeightsHidden.shape);
            Tensor loadedBiasHidden = new Tensor(modelData.BiasHidden.data, modelData.BiasHidden.shape);
            Tensor loadedWeightsOutput = new Tensor(modelData.WeightsOutput.data, modelData.WeightsOutput.shape);
            Tensor loadedBiasOutput = new Tensor(modelData.BiasOutput.data, modelData.BiasOutput.shape);

            if (loadedWeightsHidden.GetShape()[0] != modelData.InputSize ||
                loadedWeightsHidden.GetShape()[1] != modelData.HiddenSize)
                throw new Exception("Dimensões de weightsHidden não correspondem ao modelo carregado.");
            if (loadedBiasHidden.GetShape()[0] != modelData.HiddenSize)
                throw new Exception("Dimensões de biasHidden não correspondem ao modelo carregado.");
            if (loadedWeightsOutput.GetShape()[0] != modelData.HiddenSize ||
                loadedWeightsOutput.GetShape()[1] != modelData.OutputSize)
                throw new Exception("Dimensões de weightsOutput não correspondem ao modelo carregado.");
            if (loadedBiasOutput.GetShape()[0] != modelData.OutputSize)
                throw new Exception("Dimensões de biasOutput não correspondem ao modelo carregado.");

            return new NeuralNetwork(modelData.InputSize, modelData.HiddenSize, modelData.OutputSize,
                loadedWeightsHidden, loadedBiasHidden, loadedWeightsOutput, loadedBiasOutput);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar o modelo JSON (System.Text.Json): {ex.Message}");
            return null;
        }
    }
}
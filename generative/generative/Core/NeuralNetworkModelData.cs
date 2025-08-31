namespace Core;

public class NeuralNetworkModelData
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }
    // As propriedades aqui devem corresponder exatamente aos Tensors que você serializa/desserializa na NeuralNetwork
    public TensorData WeightsInputGate { get; set; } = new TensorData();
    public TensorData RecurrentWeightsInputGate { get; set; } = new TensorData();
    public TensorData BiasInputGate { get; set; } = new TensorData();
    public TensorData WeightsForgetGate { get; set; } = new TensorData();
    public TensorData RecurrentWeightsForgetGate { get; set; } = new TensorData();
    public TensorData BiasForgetGate { get; set; } = new TensorData();
    public TensorData WeightsCellGate { get; set; } = new TensorData();
    public TensorData RecurrentWeightsCellGate { get; set; } = new TensorData();
    public TensorData BiasCellGate { get; set; } = new TensorData();
    public TensorData WeightsOutputGate { get; set; } = new TensorData();
    public TensorData RecurrentWeightsOutputGate { get; set; } = new TensorData();
    public TensorData BiasOutputGate { get; set; } = new TensorData();
    public TensorData WeightsOutput { get; set; } = new TensorData(); // Weights for the final output layer
    public TensorData BiasOutput { get; set; } = new TensorData(); // Bias for the final output layer
}
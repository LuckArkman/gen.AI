namespace Core;

public class NeuralNetworkModelData
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }
    public TensorData WeightsInputGate { get; set; }
    public TensorData RecurrentWeightsInputGate { get; set; }
    public TensorData BiasInputGate { get; set; }
    public TensorData WeightsForgetGate { get; set; }
    public TensorData RecurrentWeightsForgetGate { get; set; }
    public TensorData BiasForgetGate { get; set; }
    public TensorData WeightsCellGate { get; set; }
    public TensorData RecurrentWeightsCellGate { get; set; }
    public TensorData BiasCellGate { get; set; }
    public TensorData WeightsOutputGate { get; set; }
    public TensorData RecurrentWeightsOutputGate { get; set; }
    public TensorData BiasOutputGate { get; set; }
    public TensorData WeightsOutput { get; set; }
    public TensorData BiasOutput { get; set; }
}
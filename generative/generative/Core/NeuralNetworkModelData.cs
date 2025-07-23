namespace Core;

public class NeuralNetworkModelData
{
    public int InputSize { get; set; }
    public int HiddenSize { get; set; }
    public int OutputSize { get; set; }
    public TensorData? WeightsInputGate { get; set; } // W_i
    public TensorData? RecurrentWeightsInputGate { get; set; } // U_i
    public TensorData? BiasInputGate { get; set; } // b_i
    public TensorData? WeightsForgetGate { get; set; } // W_f
    public TensorData? RecurrentWeightsForgetGate { get; set; } // U_f
    public TensorData? BiasForgetGate { get; set; } // b_f
    public TensorData? WeightsCellGate { get; set; } // W_c
    public TensorData? RecurrentWeightsCellGate { get; set; } // U_c
    public TensorData? BiasCellGate { get; set; } // b_c
    public TensorData? WeightsOutputGate { get; set; } // W_o
    public TensorData? RecurrentWeightsOutputGate { get; set; } // U_o
    public TensorData? BiasOutputGate { get; set; } // b_o
    public TensorData? WeightsOutput { get; set; } // W_out
    public TensorData? BiasOutput { get; set; } // b_out
}
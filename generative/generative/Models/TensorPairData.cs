using Core;

namespace Models;

public class TensorPairData
{
    public TensorData Input { get; set; } = new TensorData();
    public TensorData Target { get; set; } = new TensorData();
}
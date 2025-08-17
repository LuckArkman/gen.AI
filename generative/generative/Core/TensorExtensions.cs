namespace Core;

public static class TensorExtensions
{
    public static Tensor ToTensor(this TensorData data)
    {
        if (data == null || data.data == null || data.shape == null)
            throw new ArgumentNullException(nameof(data));
        return new Tensor(data.data, data.shape);
    }

    public static TensorData ToTensorData(this Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        return new TensorData { data = tensor.GetData(), shape = tensor.GetShape() };
    }
}
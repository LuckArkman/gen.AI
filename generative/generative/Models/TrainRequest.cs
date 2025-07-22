namespace Models;

public class TrainRequest
{
    public string? TextData { get; set; }
    public int SequenceLength { get; set; } = 10;
    public double? LearningRate { get; set; }
    public int? Epochs { get; set; }
}
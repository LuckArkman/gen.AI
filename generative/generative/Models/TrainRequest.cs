namespace Models;

public class TrainRequest
{
    public string? TextData { get; set; }
    public double? LearningRate { get; set; }
    public int? Epochs { get; set; }
    public int ContextWindowSize { get; set; } = 10;
}
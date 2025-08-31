namespace Models;

public class TrainRequest
{
    public string? TextData { get; set; }
    public int SequenceLength { get; set; } = 10; // Este será o ContextWindowSize
    public double? LearningRate { get; set; }
    public int? Epochs { get; set; }
    // Renomeado para ContextWindowSize para clareza sobre seu uso real
    public int ContextWindowSize { get; set; } = 10; // Novo, ou renomeado de SequenceLength
}
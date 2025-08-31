namespace Models;

public class GenerateRequest
{
    public string? SeedText { get; set; }
    public int SequenceLength { get; set; } = 10; // Renomeado para ContextWindowSize
    public int? Length { get; set; }
    public double Temperature { get; set; } = 1.0;
    public int ContextWindowSize { get; set; } = 10; // Novo, ou renomeado de SequenceLength
}
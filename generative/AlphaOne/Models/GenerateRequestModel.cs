namespace AlphaOne.Models;

public class GenerateRequestModel
{
    public string? SeedText { get; set; }
    public int SequenceLength { get; set; } = 100; // Renomeado para ContextWindowSize
    public int? Length { get; set; } = 100;
    public double Temperature { get; set; } = 1.0;
    public int ContextWindowSize { get; set; } = 10; // Novo, ou renomeado de SequenceLength
    public string? ConversationId { get; set; } = Guid.NewGuid().ToString();
}
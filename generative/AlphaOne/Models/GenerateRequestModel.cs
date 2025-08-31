namespace AlphaOne.Models;

public class GenerateRequestModel
{
    public string? SeedText { get; set; }
    public int? Length { get; set; } = 100;
    public double Temperature { get; set; } = 0.7;
    public int ContextWindowSize { get; set; } = 10;
    public string? ConversationId { get; set; }
}
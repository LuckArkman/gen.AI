namespace Models;

public class GenerateRequest
{
    public string? SeedText { get; set; }
    public int SequenceLength { get; set; } = 10;
    public int? Length { get; set; }
}
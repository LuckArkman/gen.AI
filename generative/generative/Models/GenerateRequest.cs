namespace Models;

public class GenerateRequest
{
    public string? SeedText { get; set; }
    public int SequenceLength { get; set; } // Renomeado para ContextWindowSize
    public int? Length { get; set; }
    public double Temperature { get; set; }
    public int ContextWindowSize { get; set; } // Novo, ou renomeado de SequenceLength
    public GenerateRequest(){}
    public GenerateRequest(string seedText, int sequenceLength, int? length, double temperature, int contextWindowSize)
    {
        SeedText = seedText;
        SequenceLength = sequenceLength;
        Length = length;
        Temperature = temperature;
        ContextWindowSize = contextWindowSize;
    }
}
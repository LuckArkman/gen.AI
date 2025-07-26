namespace Models;

public class GenerateRequest
{
    public string? SeedText { get; set; }
    public int? Length { get; set; }
    public double Temperature { get; set; }
    public int ContextWindowSize { get; set; } 
    public GenerateRequest(){}
    public GenerateRequest(string seedText,  int? length, double temperature, int contextWindowSize)
    {
        SeedText = seedText;
        Length = length;
        Temperature = temperature;
        ContextWindowSize = contextWindowSize;
    }
}
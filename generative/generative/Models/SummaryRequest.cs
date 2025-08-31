namespace Models;

public class SummaryRequest
{
    public string? TextToSummarize { get; set; }
    public int? SummaryLengthWords { get; set; } = 100;
    public List<string>? SourceUrls { get; set; }
}
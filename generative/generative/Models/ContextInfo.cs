namespace Models;

public class ContextInfo
{
    public string ContextId { get; set; } = string.Empty; // Inicialize
    public string Topic { get; set; } = string.Empty;     // Inicialize
    public string Summary { get; set; } = string.Empty;   // Inicialize
    public List<string> Urls { get; set; } = new List<string>(); // Inicialize
    public long ExternalLastUpdatedTicks { get; set; }
}
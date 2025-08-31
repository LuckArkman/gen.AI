using System.Text.Json.Serialization;

namespace Models;

public class Candidate
{
    // A classe 'Content' definida acima pode ser reutilizada aqui
    [JsonPropertyName("content")]
    public Content Content { get; set; }

    [JsonPropertyName("finishReason")]
    public string FinishReason { get; set; }

    [JsonPropertyName("index")]
    public int Index { get; set; }
}
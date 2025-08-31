using System.Text.Json.Serialization;

namespace Models;

public class GenerationConfig
{
    [JsonPropertyName("temperature")]
    public float Temperature { get; set; }

    [JsonPropertyName("maxOutputTokens")]
    public int MaxOutputTokens { get; set; }
}
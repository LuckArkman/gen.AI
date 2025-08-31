using System.Text.Json.Serialization;

namespace Models;

public class Part
{
    [JsonPropertyName("text")]
    public string Text { get; set; }
}
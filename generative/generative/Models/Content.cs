using System.Text.Json.Serialization;

namespace Models;

public class Content
{
    [JsonPropertyName("parts")]
    public List<Part> Parts { get; set; }
}
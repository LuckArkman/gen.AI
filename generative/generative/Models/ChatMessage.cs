using System.Text.Json.Serialization;

namespace Models;

public class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = string.Empty; // "system", "user", "assistant"

    [JsonPropertyName("content")]
    public string Content { get; set; } = string.Empty;
}
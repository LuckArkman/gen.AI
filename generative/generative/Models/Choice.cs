using System.Text.Json.Serialization;

namespace Models;

public class Choice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("message")]
    public ChatMessage? Message { get; set; }

    [JsonPropertyName("logprobs")]
    public object? LogProbs { get; set; } // Pode ser nulo ou objeto complexo

    [JsonPropertyName("finish_reason")]
    public string FinishReason { get; set; } = string.Empty;
}
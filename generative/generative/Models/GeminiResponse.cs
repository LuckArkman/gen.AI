using System.Text.Json.Serialization;

namespace Models;

public class GeminiResponse
{
    [JsonPropertyName("candidates")]
    public List<Candidate> Candidates { get; set; }
}
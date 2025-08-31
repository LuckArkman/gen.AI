namespace Models;


public class TestRequest
{
    public string? TextData { get; set; }
    public int SequenceLength { get; set; } = 10; // Renomeado para ContextWindowSize
    public int ContextWindowSize { get; set; } = 10; // Novo, ou renomeado de SequenceLength
}
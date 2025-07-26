namespace Models;

public class TestRequest
{
    public string? TextData { get; set; }
    public int ContextWindowSize { get; set; } = 10;
}
using Services;

namespace Models;

public class SaveContext
{
    public long offsetToUpdateOrInsert { get; set; }
    public byte[] serializedData { get; set; }
    public string newSummary {get; set;}
}
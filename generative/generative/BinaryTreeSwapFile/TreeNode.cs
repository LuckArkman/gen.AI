namespace BinaryTreeSwapFile;

public class TreeNode
{
    public const int MaxDataSize = 4096; // Maximum data size (bytes)
    public const int NodeSize = MaxDataSize + 8 + 8 + 8; // Data + LeftOffset + RightOffset + LastModified

    public byte[] Data { get; set; } = Array.Empty<byte>(); // Data (byte array, limited to MaxDataSize bytes)
    public long LeftOffset { get; set; } = -1; // -1 indicates no child
    public long RightOffset { get; set; } = -1; // -1 indicates no child
    public long LastModified { get; set; } = DateTime.UtcNow.Ticks; // Timestamp of last manipulation

    // Serializes the node to a fixed-size sector
    public byte[] Serialize()
    {
        byte[] buffer = new byte[NodeSize];

        // Prepare data bytes: truncate if too long, pad with zeros if too short
        byte[] dataBytes = new byte[MaxDataSize];
        Array.Copy(Data, 0, dataBytes, 0, Math.Min(Data.Length, MaxDataSize));

        // Copy to buffer
        Array.Copy(dataBytes, 0, buffer, 0, MaxDataSize);

        BitConverter.GetBytes(LeftOffset).CopyTo(buffer, MaxDataSize);
        BitConverter.GetBytes(RightOffset).CopyTo(buffer, MaxDataSize + 8);
        BitConverter.GetBytes(LastModified).CopyTo(buffer, MaxDataSize + 16);

        return buffer;
    }

    // Deserializes the node from a sector
    public static TreeNode Deserialize(byte[] buffer)
    {
        if (buffer.Length != NodeSize)
        {
            throw new ArgumentException($"Buffer must be exactly {NodeSize} bytes.");
        }

        // Extract data bytes (preserve all 32 bytes, including trailing zeros)
        byte[] data = new byte[MaxDataSize];
        Array.Copy(buffer, 0, data, 0, MaxDataSize);

        return new TreeNode
        {
            Data = data,
            LeftOffset = BitConverter.ToInt64(buffer, MaxDataSize),
            RightOffset = BitConverter.ToInt64(buffer, MaxDataSize + 8),
            LastModified = BitConverter.ToInt64(buffer, MaxDataSize + 16)
        };
    }
}
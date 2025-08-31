using System.Text;

namespace BinaryTreeSwapFile;

public class BinaryTreeFileStorage
{
    private readonly string _filePath;
    private long _rootOffset = -1; // Offset do nó raiz
    private const long RootOffsetPosition = 0; // Posição inicial para armazenar o offset da raiz
    private const int TreeDepth = 500; // Profundidade da árvore (50 camadas)

    public BinaryTreeFileStorage(string filePath)
    {
        _filePath = filePath;
        if (!File.Exists(_filePath))
        {
            // Cria o arquivo e inicializa o offset da raiz
            using (var fs = new FileStream(_filePath, FileMode.Create, FileAccess.Write))
            {
                using (var bw = new BinaryWriter(fs))
                {
                    bw.Write(_rootOffset); // Escreve -1 como offset inicial da raiz
                }
            }
        }
        else
        {
            LoadRootOffset();
        }
    }

    // Carrega o offset da raiz do início do arquivo
    private void LoadRootOffset()
    {
        using (var fs = new FileStream(_filePath, FileMode.Open, FileAccess.Read))
        {
            if (fs.Length >= sizeof(long))
            {
                using (var br = new BinaryReader(fs))
                {
                    _rootOffset = br.ReadInt64();
                }
            }
        }
    }

    private void SaveRootOffset()
    {
        using (var fs = new FileStream(_filePath, FileMode.Open, FileAccess.Write))
        {
            using (var bw = new BinaryWriter(fs))
            {
                bw.Write(_rootOffset);
            }
        }
    }

    private string GetAsString(byte[] data)
    {
        return Encoding.UTF8.GetString(data).TrimEnd('\0');
    }

    public void GenerateEmptyTree()
    {
        if (_rootOffset != -1)
        {
            Console.WriteLine("Árvore já existe. Limpe o arquivo se desejar recriar.");
            return;
        }

        long previousOffset = -1;
        for (int i = 0; i < TreeDepth; i++)
        {
            var node = new TreeNode(); // Nó vazio
            long currentOffset = WriteNewNode(node);

            if (previousOffset != -1)
            {
                // Atualiza o nó anterior para apontar para este como filho direito
                var prevNode = ReadNode(previousOffset, true);
                prevNode.RightOffset = currentOffset;
                WriteNode(previousOffset, prevNode);
            }
            else
            {
                // Define a raiz
                _rootOffset = currentOffset;
                SaveRootOffset();
            }

            previousOffset = currentOffset;
        }

        Console.WriteLine($"Árvore vazia gerada com {TreeDepth} camadas de profundidade.");
    }

    public void Insert(string data)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(data);
        if (bytes.Length > TreeNode.MaxDataSize)
            throw new ArgumentException($"Data exceeds maximum size of {TreeNode.MaxDataSize} bytes.");

        _rootOffset = InsertRecursive(_rootOffset, data);
        SaveRootOffset();
    }

    private long InsertRecursive(long offset, string data)
    {
        if (offset == -1)
        {
            // Novo nó no final do arquivo
            byte[] bytes = Encoding.UTF8.GetBytes(data);
            return WriteNewNode(new TreeNode { Data = bytes });
        }

        var node = ReadNode(offset, true);
        if (string.Compare(data, GetAsString(node.Data)) < 0)
        {
            node.LeftOffset = InsertRecursive(node.LeftOffset, data);
        }
        else if (string.Compare(data, GetAsString(node.Data)) > 0)
        {
            node.RightOffset = InsertRecursive(node.RightOffset, data);
        }
        // Se igual, não faz nada (sem duplicatas)

        // Re-escreve o nó atualizado
        WriteNode(offset, node);
        return offset;
    }

    // Busca um dado na árvore, atualizando LastModified
    public bool Search(string data)
    {
        return SearchRecursive(_rootOffset, data);
    }

    private bool SearchRecursive(long offset, string data)
    {
        if (offset == -1) return false;

        var node = ReadNode(offset, true); // Lê e atualiza LastModified
        if (string.Compare(data, GetAsString(node.Data)) == 0) return true;
        if (string.Compare(data, GetAsString(node.Data)) < 0)
            return SearchRecursive(node.LeftOffset, data);
        return SearchRecursive(node.RightOffset, data);
    }

    // Atualiza os dados de um nó específico pelo seu offset
    public void UpdateData(long offset, string newData)
    {
        byte[] newBytes = Encoding.UTF8.GetBytes(newData);
        if (newBytes.Length > TreeNode.MaxDataSize)
            throw new ArgumentException($"Data exceeds maximum size of {TreeNode.MaxDataSize} bytes.");

        var node = ReadNode(offset, true);
        node.Data = new byte[TreeNode.MaxDataSize];
        Array.Copy(newBytes, 0, node.Data, 0, newBytes.Length);
        node.LastModified = DateTime.UtcNow.Ticks; // Atualiza timestamp
        WriteNode(offset, node);
        Console.WriteLine($"Dados atualizados no nó com offset {offset}.");
    }

    // Apaga os dados de um nó específico pelo seu offset
    public void DeleteData(long offset)
    {
        var node = ReadNode(offset, true);
        node.Data = new byte[TreeNode.MaxDataSize];
        node.LastModified = DateTime.UtcNow.Ticks; // Atualiza timestamp
        WriteNode(offset, node);
        Console.WriteLine($"Dados apagados no nó com offset {offset}.");
    }

    // Obtém os dados de um nó específico pelo seu offset
    public string GetData(long offset)
    {
        var node = ReadNode(offset, true); // Atualiza LastModified via ReadNode
        return GetAsString(node.Data);
    }

    // Limpa dados de nós não manipulados por mais de um tempo limite
    public void CleanUnusedNodes(TimeSpan maxIdleTime)
    {
        CleanUnusedNodesRecursive(_rootOffset, maxIdleTime);
        Console.WriteLine("Limpeza de nós não utilizados concluída.");
    }

    private void CleanUnusedNodesRecursive(long offset, TimeSpan maxIdleTime)
    {
        if (offset == -1) return;

        var node = ReadNode(offset, false); // Lê sem atualizar LastModified
        var lastModifiedTime = new DateTime(node.LastModified);
        if (!string.IsNullOrEmpty(GetAsString(node.Data)) && DateTime.UtcNow - lastModifiedTime > maxIdleTime)
        {
            node.Data = new byte[TreeNode.MaxDataSize]; // Limpa os dados
            node.LastModified = DateTime.UtcNow.Ticks; // Atualiza timestamp
            WriteNode(offset, node);
            Console.WriteLine($"Dados limpos no nó com offset {offset}.");
        }

        CleanUnusedNodesRecursive(node.LeftOffset, maxIdleTime);
        CleanUnusedNodesRecursive(node.RightOffset, maxIdleTime);
    }

    // Travessia em ordem (para demonstração)
    public void PrintInOrder()
    {
        PrintInOrderRecursive(_rootOffset);
    }

    private void PrintInOrderRecursive(long offset)
    {
        if (offset == -1) return;

        var node = ReadNode(offset, false);
        PrintInOrderRecursive(node.LeftOffset);
        var lastModified = new DateTime(node.LastModified);
        Console.WriteLine(
            $"Offset: {offset}, Data: '{GetAsString(node.Data)}', LastModified: {lastModified:yyyy-MM-dd HH:mm:ss}");
        PrintInOrderRecursive(node.RightOffset);
    }

    // Escreve um novo nó no final do arquivo e retorna seu offset
    private long WriteNewNode(TreeNode node)
    {
        node.LastModified = DateTime.UtcNow.Ticks; // Atualiza timestamp
        using (var fs = new FileStream(_filePath, FileMode.OpenOrCreate, FileAccess.Write))
        {
            long offset = fs.Length >= sizeof(long) ? fs.Length : sizeof(long);
            fs.Seek(0, SeekOrigin.End); // Garante que estamos no final
            fs.Write(node.Serialize(), 0, TreeNode.NodeSize);
            return offset;
        }
    }

    // Re-escreve um nó em um offset específico
    private void WriteNode(long offset, TreeNode node)
    {
        node.LastModified = DateTime.UtcNow.Ticks; // Atualiza timestamp
        using (var fs = new FileStream(_filePath, FileMode.Open, FileAccess.Write))
        {
            fs.Seek(offset, SeekOrigin.Begin);
            fs.Write(node.Serialize(), 0, TreeNode.NodeSize);
        }
    }

    // Lê um nó de um offset específico
    public TreeNode ReadNode(long offset, bool updateTimestamp = true)
    {
        using (var fs = new FileStream(_filePath, FileMode.Open, FileAccess.Read))
        {
            fs.Seek(offset, SeekOrigin.Begin);
            byte[] buffer = new byte[TreeNode.NodeSize];
            int bytesRead = fs.Read(buffer, 0, TreeNode.NodeSize);
            if (bytesRead != TreeNode.NodeSize)
                throw new IOException("Falha ao ler o nó completo.");
            var node = TreeNode.Deserialize(buffer);
            if (updateTimestamp)
            {
                // Atualiza LastModified ao ler (considera leitura como manipulação)
                node.LastModified = DateTime.UtcNow.Ticks;
                WriteNode(offset, node); // Re-escreve com novo timestamp
            }

            return node;
        }
    }
    public long GetRootOffset() => _rootOffset;
}
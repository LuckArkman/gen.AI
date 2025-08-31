// Services/ContextManager.cs

using BinaryTreeSwapFile;
using System.Text;
using System.Text.Json; // Para serializar/deserializar ContextInfo
using Models; // Para a estrutura ContextInfo

namespace Services
{
    public class ContextManager
    {
        private readonly BinaryTreeFileStorage _memoryStorage;
        private readonly GoogleSearchService _googleSearchService; // <-- CORRIGIDO
        private readonly TextProcessorService _textProcessorService;

        // Definir um limite de idade para a informação, e.g., 7 dias
        public readonly TimeSpan MaxContextAgeForRefresh = TimeSpan.FromDays(7); // Tornar public/readonly

        // Limite para considerar similaridade (pode ser ajustado)
        private readonly double _similarityThreshold = 0.7; // Exemplo para uma métrica futura

        public ContextManager(BinaryTreeFileStorage memoryStorage, 
            GoogleSearchService webSearchService, 
            TextProcessorService textProcessorService)
        {
            _memoryStorage = memoryStorage;
            _googleSearchService = webSearchService; // Atribuir
            _textProcessorService = textProcessorService;
        }
        
        public async Task StoreConversationContext(string conversationInput, string conversationResponse)
        {
            string fullContext = $"{conversationInput} {conversationResponse}";
            string topic = _textProcessorService.ExtractMainTopic(fullContext);
            string summary = _textProcessorService.Summarize(fullContext, TreeNode.MaxDataSize / 2); // Metade do tamanho máximo

            ContextInfo currentContextInfo = new ContextInfo 
            { 
                ContextId = _textProcessorService.GenerateContextHash(topic), 
                Topic = topic, 
                Summary = summary, 
                ExternalLastUpdatedTicks = DateTime.UtcNow.Ticks,
                Urls = new List<string>() // A conversa não tem URLs de fonte externa
            };

            byte[] serializedData = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(currentContextInfo));
            if (serializedData.Length > TreeNode.MaxDataSize)
            {
                Console.WriteLine($"Aviso: Contexto de conversa muito grande ({serializedData.Length} bytes). Truncando para {TreeNode.MaxDataSize} bytes antes de armazenar.");
                Array.Resize(ref serializedData, TreeNode.MaxDataSize);
            }
            
            long existingOffset = FindContextOffsetByTopic(topic); 
            if (existingOffset != -1)
            {
                Console.WriteLine($"Contexto de conversa similar encontrado para '{topic}'. Atualizando dados...");
                _memoryStorage.UpdateData(existingOffset, Encoding.UTF8.GetString(serializedData));
            }
            else
            {
                Console.WriteLine($"Novo contexto de conversa para '{topic}'. Inserindo na memória virtual...");
                _memoryStorage.Insert(Encoding.UTF8.GetString(serializedData));
            }
            _memoryStorage.CleanUnusedNodes(TimeSpan.FromDays(30));
        }

        public async Task StoreContext(string conversationContext, string response)
        {
            // Concatenar a conversa para formar o contexto principal a ser armazenado
            string fullContext = $"{conversationContext} {response}";
            string topic = _textProcessorService.ExtractMainTopic(fullContext);
            string summary =
                _textProcessorService.Summarize(fullContext, TreeNode.MaxDataSize / 2); // Resumo para armazenamento
            string contextId = _textProcessorService.GenerateContextHash(topic); // Usar tópico como ID para busca

            // Tentar encontrar um contexto similar na memória virtual
            long existingOffset = FindContextOffsetByTopic(topic); // Precisaremos de uma busca mais inteligente

            ContextInfo currentContextInfo = new ContextInfo
            {
                ContextId = contextId,
                Topic = topic,
                Summary = summary,
                ExternalLastUpdatedTicks = DateTime.UtcNow.Ticks,
                Urls = new List<string>()
            };

            byte[] serializedData = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(currentContextInfo));
            if (serializedData.Length > TreeNode.MaxDataSize)
            {
                Console.WriteLine(
                    $"Aviso: Contexto muito grande ({serializedData.Length} bytes). Truncando para {TreeNode.MaxDataSize} bytes antes de armazenar.");
                Array.Resize(ref serializedData, TreeNode.MaxDataSize);
            }

            // Para a BST atual, vamos "inserir" ou "atualizar" o nó usando o tópico como chave primária.
            // Isso significa que se o tópico já existe, ele será sobrescrito.
            // Uma árvore binária de busca não é ideal para "similaridade", mas serve para chave exata.
            if (existingOffset != -1)
            {
                Console.WriteLine($"Contexto similar encontrado para '{topic}'. Atualizando dados...");
                _memoryStorage.UpdateData(existingOffset,
                    Encoding.UTF8.GetString(serializedData)); // UpdateData espera string
            }
            else
            {
                Console.WriteLine($"Novo contexto para '{topic}'. Inserindo na memória virtual...");
                _memoryStorage.Insert(Encoding.UTF8.GetString(serializedData)); // Insert espera string
            }

            _memoryStorage.CleanUnusedNodes(TimeSpan.FromDays(30)); // Limpar nós antigos
        }

        public async Task<string> ReflectAndEnrich(string currentInput)
        {
            string topic = _textProcessorService.ExtractMainTopic(currentInput);
            ContextInfo? storedContext = GetContextByTopic(topic);
            string enrichedInformation = string.Empty;

            bool needsWebSearch = false;

            if (storedContext == null)
            {
                Console.WriteLine(
                    $"Reflexão: Nenhum contexto para '{topic}' na memória. Iniciando busca na internet para nova informação.");
                needsWebSearch = true;
            }
            else
            {
                DateTime lastUpdated = new DateTime(storedContext.ExternalLastUpdatedTicks);
                if (DateTime.UtcNow - lastUpdated > MaxContextAgeForRefresh)
                {
                    Console.WriteLine(
                        $"Reflexão: Contexto para '{topic}' desatualizado. Buscando na internet para atualização.");
                    needsWebSearch = true;
                }
                else
                {
                    // Contexto existente e atualizado. Usar o que já temos.
                    enrichedInformation = storedContext.Summary;
                    Console.WriteLine($"Contexto recuperado da memória virtual (atualizado): {storedContext.Summary}");
                }
            }

            if (needsWebSearch)
            {
                var webContent = await _googleSearchService.SearchAndExtractText(topic);
                if (webContent.Any())
                {
                    string newSummary = _textProcessorService.Summarize(string.Join(" ", webContent));

                    ContextInfo contextToStore = storedContext ?? new ContextInfo // Use o existente ou crie um novo
                    {
                        ContextId = _textProcessorService.GenerateContextHash(topic),
                        Topic = topic,
                        Urls = new List<string>()
                    };

                    // Só atualiza se a nova informação for diferente da armazenada (ou se não havia nada antes)
                    if (storedContext == null || !string.Equals(newSummary.Trim(), storedContext.Summary.Trim(),
                            StringComparison.OrdinalIgnoreCase))
                    {
                        Console.WriteLine(
                            "Informação relevante encontrada na internet. Atualizando/Inserindo memória virtual.");
                        contextToStore.Summary = newSummary;
                        contextToStore.ExternalLastUpdatedTicks = DateTime.UtcNow.Ticks;
                        // Simular adição de URLs de fontes
                        contextToStore.Urls = webContent.Select((_, i) =>
                            $"http://example.com/source_{i + 1}_{topic.Replace(" ", "_")}.com").ToList();

                        byte[] serializedData = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(contextToStore));
                        if (serializedData.Length > TreeNode.MaxDataSize)
                        {
                            Console.WriteLine(
                                $"Aviso: Contexto da web muito grande ({serializedData.Length} bytes). Truncando para {TreeNode.MaxDataSize} bytes antes de armazenar.");
                            Array.Resize(ref serializedData, TreeNode.MaxDataSize);
                        }

                        // A lógica de Insert/Update no BinaryTreeFileStorage atualmente usa string como chave,
                        // então estamos passando a string serializada do ContextInfo.
                        // Isso significa que FindContextOffsetByTopic precisa ser preciso na sua busca.
                        long offsetToUpdateOrInsert = FindContextOffsetByTopic(topic);
                        if (offsetToUpdateOrInsert != -1)
                        {
                            _memoryStorage.UpdateData(offsetToUpdateOrInsert, Encoding.UTF8.GetString(serializedData));
                        }
                        else
                        {
                            _memoryStorage.Insert(Encoding.UTF8.GetString(serializedData));
                        }

                        _memoryStorage.CleanUnusedNodes(TimeSpan.FromDays(30)); // Manter limpeza
                    }
                    else
                    {
                        Console.WriteLine(
                            "Informação da internet não é significativamente diferente ou é redundante. Não atualizando o contexto existente.");
                        // Mesmo se não atualizar, use o contexto existente
                        if (storedContext != null) enrichedInformation = storedContext.Summary;
                    }

                    // Usar a nova (ou a recém-confirmada) informação para enriquecimento
                    enrichedInformation =
                        newSummary; // Use a novaSummary, mesmo que não tenha atualizado o disco, ela é a mais recente
                }
                else
                {
                    Console.WriteLine("Busca na internet não retornou conteúdo relevante.");
                    // Se não encontrou na internet, mas tinha algo antigo, use o antigo
                    if (storedContext != null) enrichedInformation = storedContext.Summary;
                }
            }

            return
                enrichedInformation; // Retorna o contexto enriquecido (pode ser vazio se nada foi encontrado/atualizado)
        }

        // Helpers para a integração com BinaryTreeFileStorage
        public long FindContextOffsetByTopic(string topic)
        {
            long foundOffset = -1;
            // GetRootOffset() é necessário para iniciar a busca
            long rootOffset = _memoryStorage.GetRootOffset(); 
            TraverseAndFindOffset(rootOffset, topic, ref foundOffset);
            return foundOffset;
        }

        public ContextInfo? GetContextByTopic(string topic)
        {
            long offset = FindContextOffsetByTopic(topic);
            if (offset != -1)
            {
                try
                {
                    string rawData = _memoryStorage.GetData(offset); 
                    ContextInfo? info = JsonSerializer.Deserialize<ContextInfo>(rawData);
                    return info;
                }
                catch (JsonException ex)
                {
                    Console.WriteLine($"Erro ao deserializar contexto do offset {offset}: {ex.Message}");
                    return null;
                }
            }
            return null;
        }

        // Método auxiliar para FindContextOffsetByTopic
        private void TraverseAndFindOffset(long currentOffset, string targetTopic, ref long foundOffset)
        {
            if (currentOffset == -1 || foundOffset != -1) return;

            // ReadNode aqui não deve atualizar o timestamp, apenas ler
            TreeNode node = _memoryStorage.ReadNode(currentOffset, false); 
            string rawData = Encoding.UTF8.GetString(node.Data).TrimEnd('\0');

            if (!string.IsNullOrEmpty(rawData))
            {
                try
                {
                    ContextInfo? info = JsonSerializer.Deserialize<ContextInfo>(rawData);
                    if (info != null && info.Topic.Equals(targetTopic, StringComparison.OrdinalIgnoreCase))
                    {
                        foundOffset = currentOffset;
                        return;
                    }
                }
                catch (JsonException)
                {
                    // Ignorar nós inválidos
                }
            }

            TraverseAndFindOffset(node.LeftOffset, targetTopic, ref foundOffset);
            TraverseAndFindOffset(node.RightOffset, targetTopic, ref foundOffset);
        }
    }
}
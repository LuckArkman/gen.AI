using System.Text;
using System.Text.Json; // Usar System.Text.Json para ContextInfo
using System.Text.RegularExpressions;
using BinaryTreeSwapFile;
using Models; // Assumindo ContextInfo, SummaryRequest, etc. estão aqui
using Services; // Namespace correto para TextProcessorService e ContextInfo

// Manter para consistência se no futuro usar IConfiguration aqui.

namespace Core
{
    public class Trainer
    {
        private readonly string datasetPath;
        private readonly string modelPathTemplate;
        private readonly string _vocabPath;
        private NeuralNetwork? model;
        private Dictionary<string, int> tokenToIndex;
        private List<string> indexToToken;
        private readonly int hiddenSize;
        private readonly int contextWindowSize; 
        private readonly double learningRate;
        private readonly int epochs;
        private readonly string padToken = "[PAD]";
        private readonly string logPath; // Necessário para logar treinamento
        private readonly TextProcessorService _textProcessorService;
        private readonly BinaryTreeFileStorage _memoryStorage;
        private readonly int _knowledgeSummaryLength = 200;

        public Trainer(string datasetPath, string modelPathTemplate, string vocabPath,
            int hiddenSize, int sequenceLength, double learningRate, int epochs,
            TextProcessorService textProcessorService, BinaryTreeFileStorage memoryStorage)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            if (string.IsNullOrEmpty(datasetPath))
                throw new ArgumentNullException(nameof(datasetPath));
            if (string.IsNullOrEmpty(modelPathTemplate))
                throw new ArgumentNullException(nameof(modelPathTemplate));
            if (string.IsNullOrEmpty(vocabPath))
                throw new ArgumentNullException(nameof(vocabPath));
            if (sequenceLength <= 0)
                throw new ArgumentException("ContextWindowSize deve ser positivo.", nameof(sequenceLength));
            if (learningRate <= 0)
                throw new ArgumentException("LearningRate deve ser positivo.", nameof(learningRate));
            if (epochs <= 0)
                throw new ArgumentException("Epochs deve ser positivo.", nameof(epochs));

            this.datasetPath = datasetPath;
            this.modelPathTemplate = modelPathTemplate;
            this._vocabPath = vocabPath;
            this.hiddenSize = hiddenSize;
            this.contextWindowSize = sequenceLength;
            this.learningRate = learningRate;
            this.epochs = epochs;
            this.logPath = Path.Combine(Path.GetDirectoryName(datasetPath) ?? "", "training_log.txt"); // Inicializa logPath

            tokenToIndex = new Dictionary<string, int>();
            indexToToken = new List<string>();
            _textProcessorService = textProcessorService ?? throw new ArgumentNullException(nameof(textProcessorService));
            _memoryStorage = memoryStorage ?? throw new ArgumentNullException(nameof(memoryStorage));
        }

        public void Train(int startEpoch = 1, int chunkSize = 1000) // Reintroduzindo chunkSize
        {
            try
            {
                if (!File.Exists(datasetPath))
                {
                    throw new FileNotFoundException($"Arquivo do dataset não encontrado: {datasetPath}");
                }

                if (startEpoch <= 0)
                {
                    throw new ArgumentException("startEpoch deve ser positivo.", nameof(startEpoch));
                }
                if (chunkSize <= 0) // Validação para chunkSize
                {
                    throw new ArgumentException("chunkSize deve ser positivo.", nameof(chunkSize));
                }

                ValidateFileEncoding();

                if (File.Exists(_vocabPath))
                {
                    Console.WriteLine($"Tentando carregar vocabulário existente de: {_vocabPath}");
                    LoadVocabulary(_vocabPath);
                    if (tokenToIndex.Count <= 1)
                    {
                        Console.WriteLine("Vocabulário vazio ou inválido. Construindo vocabulário do dataset.");
                        BuildVocabularyFromDataset();
                    }
                }
                else
                {
                    Console.WriteLine("Nenhum vocabulário encontrado. Construindo vocabulário do dataset.");
                    BuildVocabularyFromDataset();
                }

                if (tokenToIndex.Count <= 1)
                {
                    throw new InvalidOperationException("Nenhum token válido encontrado no dataset para construir o vocabulário (além do token de padding).");
                }

                string modelToLoadPath = modelPathTemplate; // Padrão se não for retomar
                if (startEpoch > 1)
                {
                    // A ideia era carregar o modelo da época ANTERIOR, para continuar.
                    // modelPathTemplate é "model.json", então não tem {epoch}
                    // Apenas model.json será carregado. Ou, se for por época, o trainer deve nomear assim.
                    // Vamos simplificar e carregar sempre de 'model.json' se existir.
                    modelToLoadPath = modelPathTemplate; // Sempre o mesmo arquivo se o modelo não nomeia por época
                }
                
                if (File.Exists(modelToLoadPath))
                {
                    Console.WriteLine($"Tentando carregar modelo de: {modelToLoadPath}...");
                    model = NeuralNetwork.LoadModel(modelToLoadPath);
                    if (model == null)
                    {
                        Console.WriteLine($"Falha ao carregar modelo previo. Inicializando novo modelo.");
                        model = new NeuralNetwork(tokenToIndex.Count * contextWindowSize, hiddenSize, tokenToIndex.Count, contextWindowSize); // CORRIGIDO: 4 argumentos
                    }
                    else if (model.InputSize != tokenToIndex.Count * contextWindowSize || model.OutputSize != tokenToIndex.Count)
                    {
                        Console.WriteLine($"Tamanho do vocabulário ({tokenToIndex.Count}) ou ContextWindowSize ({contextWindowSize}) não corresponde ao modelo carregado (Input: {model.InputSize}, Output: {model.OutputSize}). Inicializando novo modelo.");
                        model = new NeuralNetwork(tokenToIndex.Count * contextWindowSize, hiddenSize, tokenToIndex.Count, contextWindowSize); // CORRIGIDO: 4 argumentos
                    }
                    else
                    {
                         Console.WriteLine($"Modelo Previo carregado com sucesso.");
                    }
                }
                else
                {
                    Console.WriteLine("Nenhum modelo anterior encontrado ou iniciando do zero. Inicializando novo modelo.");
                    model = new NeuralNetwork(tokenToIndex.Count * contextWindowSize, hiddenSize, tokenToIndex.Count, contextWindowSize); // CORRIGIDO: 4 argumentos
                }

                if (model == null)
                {
                     throw new InvalidOperationException("Falha ao inicializar o modelo de rede neural.");
                }

                for (int epoch = startEpoch; epoch <= epochs; epoch++)
                {
                    Console.WriteLine($"Iniciando época {epoch}/{epochs}");
                    double totalLoss = 0;
                    int chunkCount = 0;

                    using (var reader = new StreamReader(datasetPath, Encoding.UTF8, true))
                    {
                        bool endOfFile = false;
                        while (!endOfFile)
                        {
                            var lines = new List<string>(chunkSize);
                            for (int i = 0; i < chunkSize; i++)
                            {
                                string? line = reader.ReadLine();
                                if (line == null)
                                {
                                    endOfFile = true;
                                    break;
                                }
                                if (!string.IsNullOrEmpty(line))
                                {
                                    lines.Add(line);
                                }
                            }

                            if (lines.Count > 0)
                            {
                                chunkCount++;
                                string chunkText = string.Join("\n", lines);
                                ProcessChunk(chunkText, ref totalLoss, chunkCount, epoch); // CORRIGIDO: Passa 'epoch'
                                GC.Collect();
                            }
                        }
                    }

                    if (chunkCount == 0)
                    {
                        Console.WriteLine("Nenhum chunk válido encontrado no dataset para esta época. Verifique o arquivo de entrada.");
                        continue; 
                    }

                    double averageLoss = totalLoss / chunkCount;
                    var logMessage = $"Época {epoch}/{epochs} concluída. Perda média: {averageLoss:F4}, Total de chunks processados: {chunkCount}, Tamanho do vocabulário: {tokenToIndex.Count}";
                    Console.WriteLine(logMessage);
                    File.AppendAllText(logPath, logMessage + "\n");
                    // Salva o modelo no final de cada época (ou em intervalos, dependendo da estratégia)
                    model.SaveModel(modelPathTemplate); // Salva no arquivo padrão 'model.json'
                }

                // A versão final já está salva no loop da época
                // string finalModelPath = modelPathTemplate.Replace("{epoch}", epochs.ToString());
                // try { model.SaveModel(finalModelPath); } catch (Exception ex) { Console.WriteLine($"Aviso: Falha ao salvar o modelo final ({finalModelPath}): {ex.Message}"); }

                Console.WriteLine("Treinamento concluído.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro crítico durante o treinamento: {ex.Message}");
                throw;
            }
        }
        
        private void ValidateFileEncoding()
        {
            try
            {
                using (var reader = new StreamReader(datasetPath, Encoding.UTF8, true))
                {
                    char[] buffer = new char[4096];
                    int charsRead;
                    while ((charsRead = reader.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        for (int i = 0; i < charsRead; i++)
                        {
                            if (buffer[i] == '\uFFFD')
                            {
                                throw new InvalidOperationException("O arquivo contém caracteres inválidos (substituição \\uFFFD). Verifique a codificação do arquivo.");
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao validar a codificação do arquivo: {ex.Message}");
                throw;
            }
        }

        private void BuildVocabularyFromDataset()
        {
            tokenToIndex.Clear();
            indexToToken.Clear();
            tokenToIndex[padToken] = indexToToken.Count;
            indexToToken.Add(padToken);

            var specialChars = new[] {
                '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*',
                '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
                '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|',
                '}', '~'
            };
            var specialCharPattern = string.Join("|",
                specialChars.Select(c => Regex.Escape(c.ToString()))
            );
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";

            HashSet<string> uniqueTokens = new HashSet<string>();

            using (var reader = new StreamReader(datasetPath, Encoding.UTF8, true))
            {
                int lineNumber = 0;
                while (!reader.EndOfStream)
                {
                    lineNumber++;
                    string? line = reader.ReadLine();
                    if (string.IsNullOrEmpty(line)) continue;

                    string normalizedLine = line.ToLower();
                    var matches = Regex.Matches(normalizedLine, pattern);
                    foreach (Match match in matches)
                    {
                        string token = match.Value;
                        if (char.IsControl(token[0]) && token[0] != ' ' || token == "\uFFFD" || (int)token[0] > 0x10FFFF)
                        {
                            Console.WriteLine($"Token inválido '{token}' ignorado na linha {lineNumber} durante a construção do vocabulário.");
                            continue;
                        }
                        uniqueTokens.Add(token);
                    }
                }
            }

            var sortedTokens = uniqueTokens.OrderBy(t => t).ToList();
            foreach (string token in sortedTokens)
            {
                if (!tokenToIndex.ContainsKey(token))
                {
                    tokenToIndex[token] = indexToToken.Count;
                    indexToToken.Add(token);
                }
            }

            if (tokenToIndex.Count <= 1)
            {
                throw new InvalidOperationException("Nenhum token válido (além do token de padding) encontrado no dataset para construir o vocabulário.");
            }

            Console.WriteLine($"Vocabulário inicial construído. Tamanho: {tokenToIndex.Count} tokens.");
            SaveVocabulary(_vocabPath);
        }

        public bool LoadModelAndVocabulary(string modelPath, string vocabPath)
        {
            try
            {
                LoadVocabulary(vocabPath);
                if (tokenToIndex.Count <= 1)
                {
                    Console.WriteLine($"Falha ao carregar o vocabulário de: {vocabPath}. Vocabulário vazio ou com apenas o token [PAD].");
                    return false;
                }

                model = NeuralNetwork.LoadModel(modelPath);
                if (model == null)
                {
                    Console.WriteLine($"Falha ao carregar o modelo de: {modelPath}.");
                    return false;
                }

                if (model.InputSize != tokenToIndex.Count * contextWindowSize || model.OutputSize != tokenToIndex.Count)
                {
                    Console.WriteLine($"Modelo carregado de {modelPath} não corresponde ao tamanho do vocabulário ({tokenToIndex.Count}) OU ContextWindowSize ({contextWindowSize}). " +
                                      $"Modelo: Input {model.InputSize}, Output {model.OutputSize}. Vocabulário x Janela: {tokenToIndex.Count * contextWindowSize}.");
                    model = null;
                    return false;
                }

                Console.WriteLine($"Modelo e vocabulário carregados com sucesso de: {modelPath}, {vocabPath}");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar modelo ou vocabulário: {ex.Message}");
                model = null;
                tokenToIndex.Clear();
                indexToToken.Clear();
                tokenToIndex[padToken] = indexToToken.Count;
                indexToToken.Add(padToken);
                return false;
            }
        }

        private void LoadVocabulary(string vocabPath)
        {
            tokenToIndex.Clear();
            indexToToken.Clear();
            tokenToIndex[padToken] = indexToToken.Count;
            indexToToken.Add(padToken);

            try
            {
                using (var reader = new StreamReader(vocabPath, Encoding.UTF8, true))
                {
                    int lineNumber = 0;
                    while (!reader.EndOfStream)
                    {
                        lineNumber++;
                        string line = reader.ReadLine()!.Trim();
                        if (string.IsNullOrEmpty(line)) continue;

                        string token = line;
                        if (token == padToken && tokenToIndex.ContainsKey(padToken)) continue;

                        if (char.IsControl(token[0]) && token[0] != ' ' || token == "\uFFFD" || (int)token[0] > 0x10FFFF)
                        {
                            Console.WriteLine($"Token inválido '{token}' ignorado no vocabulário na linha {lineNumber} durante o carregamento.");
                            continue;
                        }

                        if (!tokenToIndex.ContainsKey(token))
                        {
                            tokenToIndex[token] = indexToToken.Count;
                            indexToToken.Add(token);
                        }
                    }
                }

                if (indexToToken.Count <= 1 && tokenToIndex.ContainsKey(padToken))
                {
                    throw new InvalidOperationException("Nenhum token válido encontrado no arquivo de vocabulário (além do token de padding, se presente).");
                }
                else if (indexToToken.Count == 0)
                {
                    throw new InvalidOperationException("Nenhum token encontrado no arquivo de vocabulário.");
                }

                Console.WriteLine($"Vocabulário carregado de: {vocabPath}, Tamanho: {tokenToIndex.Count} tokens.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar vocabulário: {ex.Message}");
                tokenToIndex = new Dictionary<string, int>();
                indexToToken = new List<string>();
                tokenToIndex[padToken] = indexToToken.Count;
                indexToToken.Add(padToken);
                return;
            }
        }

        // CORRIGIDO: Adicionado parâmetro 'epoch'
        private void ProcessChunk(string chunkText, ref double totalLoss, int chunkIndex, int epoch)
        {
            if (string.IsNullOrEmpty(chunkText))
            {
                Console.WriteLine($"Chunk {chunkIndex} ignorado: chunk vazio.");
                return;
            }
            // CORRIGIDO: PrepareDataset agora aceita contextWindowSize
            var (inputs, targets) = PrepareDataset(chunkText, contextWindowSize); 
            if (inputs.Length == 0 || targets.Length == 0)
            {
                Console.WriteLine($"Chunk {chunkIndex} processado, mas não gerou dados de treinamento válidos (sequências insuficientes ou tokens ausentes no vocabulário).");
                return;
            }

            if (model == null)
            {
                throw new InvalidOperationException($"Modelo não inicializado para o chunk {chunkIndex}. Isso não deveria acontecer após o setup inicial.");
            }

            if (model.InputSize != tokenToIndex.Count * contextWindowSize || model.OutputSize != tokenToIndex.Count)
            {
                Console.WriteLine($"Erro: Modelo não corresponde ao tamanho do vocabulário fixo ({tokenToIndex.Count}) ou ContextWindowSize ({contextWindowSize}). " +
                                  $"Modelo: InputSize={model.InputSize}, Expected InputSize={tokenToIndex.Count * contextWindowSize}, OutputSize={model.OutputSize}.");
                throw new InvalidOperationException("Incompatibilidade entre o modelo e o vocabulário fixo/ContextWindowSize durante o treinamento.");
            }

            double chunkLoss = model.TrainEpoch(inputs, targets, learningRate);
            totalLoss += chunkLoss;

            // --- NOVO: Armazenar conhecimento do chunk na memória virtual ---
            if (epoch == 1 ) // Exemplo: Armazenar a cada 10 chunks na primeira época
            {
                Console.WriteLine($"Armazenando resumo do chunk {chunkIndex} na memória virtual...");
                string chunkTopic = _textProcessorService.ExtractMainTopic(chunkText);
                string chunkSummary = _textProcessorService.Summarize(chunkText, _knowledgeSummaryLength);

                ContextInfo chunkKnowledge = new ContextInfo
                {
                    ContextId = _textProcessorService.GenerateContextHash(chunkTopic),
                    Topic = chunkTopic,
                    Summary = chunkSummary,
                    Urls = new List<string> { datasetPath }, // Fonte é o dataset
                    ExternalLastUpdatedTicks = DateTime.UtcNow.Ticks
                };

                byte[] serializedData = Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(chunkKnowledge));
                if (serializedData.Length > TreeNode.MaxDataSize)
                {
                    Console.WriteLine($"Aviso: Conhecimento do chunk muito grande ({serializedData.Length} bytes). Truncando para {TreeNode.MaxDataSize} bytes.");
                    Array.Resize(ref serializedData, TreeNode.MaxDataSize);
                }
                
                try
                {
                     _memoryStorage.Insert(Encoding.UTF8.GetString(serializedData));
                     Console.WriteLine($"Conhecimento do chunk armazenado/atualizado na memória virtual.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Erro ao armazenar conhecimento do chunk: {ex.Message}");
                }
            }

            string logMessage = $"Época {epoch}/{epochs}, Chunk {chunkIndex} processado, Perda: {chunkLoss:F4}";
            Console.WriteLine(logMessage);
            File.AppendAllText(logPath, logMessage + "\n");
        }

        // CORRIGIDO: Adicionado parâmetro 'currentContextWindowSize'
        private (Tensor[] inputs, Tensor[] targets) PrepareDataset(string text, int currentContextWindowSize)
        {
            var inputs = new List<Tensor>();
            var targets = new List<Tensor>();

            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            
            var matches = Regex.Matches(text.ToLower(), pattern);
            var tokens = matches.Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToArray();

            var paddedTokens = new List<string>();
            for (int k = 0; k < currentContextWindowSize; k++)
            {
                paddedTokens.Add(padToken);
            }
            paddedTokens.AddRange(tokens);

            for (int i = 0; i < paddedTokens.Count - currentContextWindowSize; i++)
            {
                string[] currentWindowTokens = paddedTokens.Skip(i).Take(currentContextWindowSize).ToArray();
                string nextToken = paddedTokens[i + currentContextWindowSize];

                if (!tokenToIndex.ContainsKey(nextToken) || !currentWindowTokens.All(t => tokenToIndex.ContainsKey(t)))
                {
                    Console.WriteLine($"Sequência ignorada no dataset (índice {i}): tokens ausentes no vocabulário fixo. Token de predição: '{nextToken}', Sequência de entrada: '{string.Join(" ", currentWindowTokens)}'");
                    continue;
                }

                double[] inputData = new double[tokenToIndex.Count * currentContextWindowSize];
                for (int k = 0; k < currentContextWindowSize; k++)
                {
                    int tokenVocabIndex = tokenToIndex[currentWindowTokens[k]];
                    int offset = k * tokenToIndex.Count;
                    inputData[offset + tokenVocabIndex] = 1.0;
                }
                inputs.Add(new Tensor(inputData, [tokenToIndex.Count * currentContextWindowSize]));

                double[] targetData = new double[tokenToIndex.Count];
                targetData[tokenToIndex[nextToken]] = 1.0;
                targets.Add(new Tensor(targetData, new int[] { tokenToIndex.Count }));
            }

            return (inputs.ToArray(), targets.ToArray());
        }

        private void SaveVocabulary(string vocabPath)
        {
            try
            {
                using (var writer = new StreamWriter(vocabPath, false, new UTF8Encoding(false)))
                {
                    foreach (string token in indexToToken)
                    {
                        writer.WriteLine(token);
                    }
                }
                Console.WriteLine($"Vocabulário salvo em: {vocabPath}, Tamanho: {tokenToIndex.Count} tokens.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar vocabulário: {ex.Message}");
                throw;
            }
        }
    }
}
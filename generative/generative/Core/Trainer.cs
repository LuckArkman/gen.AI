using System.Text;
using System.Text.RegularExpressions;

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
        private readonly int contextWindowSize; // Novo: Tamanho da janela de contexto
        private readonly double learningRate;
        private readonly int epochs;
        private readonly string padToken = "[PAD]";

        public Trainer(string datasetPath, string modelPathTemplate, string vocabPath,
            int hiddenSize = 256, int sequenceLength = 10, double learningRate = 0.01, int epochs = 10)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            if (string.IsNullOrEmpty(datasetPath))
                throw new ArgumentNullException(nameof(datasetPath));
            if (string.IsNullOrEmpty(modelPathTemplate))
                throw new ArgumentNullException(nameof(modelPathTemplate));
            if (string.IsNullOrEmpty(vocabPath))
                throw new ArgumentNullException(nameof(vocabPath));
            if (sequenceLength <= 0) // Agora é contextWindowSize
                throw new ArgumentException("ContextWindowSize deve ser positivo.", nameof(sequenceLength));
            if (learningRate <= 0)
                throw new ArgumentException("LearningRate deve ser positivo.", nameof(learningRate));
            if (epochs <= 0)
                throw new ArgumentException("Epochs deve ser positivo.", nameof(epochs));

            this.datasetPath = datasetPath;
            this.modelPathTemplate = modelPathTemplate;
            this._vocabPath = vocabPath;
            this.hiddenSize = hiddenSize;
            this.contextWindowSize = sequenceLength; // Atribui SequenceLength ao novo ContextWindowSize
            this.learningRate = learningRate;
            this.epochs = epochs;

            tokenToIndex = new Dictionary<string, int>();
            indexToToken = new List<string>();
        }

        public void Train(int startEpoch = 1)
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

                // Ajusta modelPath para carregar a última época treinada se startEpoch > 1
                string modelToLoadPath = modelPathTemplate.Replace("{epoch}", (startEpoch - 1).ToString());
                
                if (startEpoch > 1 && File.Exists(modelToLoadPath))
                {
                    Console.WriteLine($"Tentando carregar modelo da época {startEpoch - 1} de: {modelToLoadPath}...");
                    model = NeuralNetwork.LoadModel(modelToLoadPath);
                    if (model == null)
                    {
                        Console.WriteLine($"Falha ao carregar modelo da época {startEpoch - 1}. Inicializando novo modelo.");
                        // Input size agora é vocabSize * contextWindowSize
                        model = new NeuralNetwork(tokenToIndex.Count * contextWindowSize, hiddenSize, tokenToIndex.Count);
                    }
                    else if (model.InputSize != tokenToIndex.Count * contextWindowSize || model.OutputSize != tokenToIndex.Count)
                    {
                        Console.WriteLine($"Tamanho do vocabulário ({tokenToIndex.Count}) ou ContextWindowSize ({contextWindowSize}) não corresponde ao modelo carregado (Input: {model.InputSize}, Output: {model.OutputSize}). Inicializando novo modelo.");
                        model = new NeuralNetwork(tokenToIndex.Count * contextWindowSize, hiddenSize, tokenToIndex.Count);
                    }
                    else
                    {
                         Console.WriteLine($"Modelo da época {startEpoch - 1} carregado com sucesso.");
                    }
                }
                else
                {
                    Console.WriteLine("Nenhum modelo anterior encontrado ou iniciando do zero. Inicializando novo modelo.");
                    // Input size agora é vocabSize * contextWindowSize
                    model = new NeuralNetwork(tokenToIndex.Count * contextWindowSize, hiddenSize, tokenToIndex.Count);
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
                        int lineNumber = 0;
                        while (!reader.EndOfStream)
                        {
                            lineNumber++;
                            string? line = reader.ReadLine();
                            if (string.IsNullOrEmpty(line))
                            {
                                Console.WriteLine($"Linha {lineNumber} ignorada: linha vazia.");
                                continue;
                            }

                            chunkCount++;
                            ProcessChunk(line, ref totalLoss, chunkCount); 
                            GC.Collect(); 
                        }
                    }

                    if (chunkCount == 0)
                    {
                        Console.WriteLine("Nenhum chunk válido encontrado no dataset para esta época. Verifique o arquivo de entrada.");
                        continue; 
                    }

                    double averageLoss = totalLoss / chunkCount;
                    Console.WriteLine($"Época {epoch}/{epochs} concluída. Perda média: {averageLoss:F4}, Total de chunks processados: {chunkCount}, Tamanho do vocabulário: {tokenToIndex.Count}");
                }

                string finalModelPath = modelPathTemplate.Replace("{epoch}", epochs.ToString());
                try
                {
                    model.SaveModel(finalModelPath);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Aviso: Falha ao salvar o modelo final ({finalModelPath}): {ex.Message}");
                }

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

                // Crucialmente, verifica se as dimensões do modelo carregado correspondem ao vocabulário e à janela de contexto
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

        private void ProcessChunk(string chunkText, ref double totalLoss, int chunkIndex)
        {
            if (string.IsNullOrEmpty(chunkText))
            {
                Console.WriteLine($"Chunk {chunkIndex} ignorado: chunk vazio.");
                return;
            }

            var (inputs, targets) = PrepareDataset(chunkText);
            if (inputs.Length == 0 || targets.Length == 0)
            {
                Console.WriteLine($"Chunk {chunkIndex} processado, mas não gerou dados de treinamento válidos (sequências insuficientes ou tokens ausentes no vocabulário).");
                return;
            }

            if (model == null)
            {
                throw new InvalidOperationException($"Modelo não inicializado para o chunk {chunkIndex}. Isso não deveria acontecer após o setup inicial.");
            }

            // Verifica compatibilidade do modelo com o vocabulário e a janela de contexto
            if (model.InputSize != tokenToIndex.Count * contextWindowSize || model.OutputSize != tokenToIndex.Count)
            {
                Console.WriteLine($"Erro: Modelo não corresponde ao tamanho do vocabulário fixo ({tokenToIndex.Count}) ou ContextWindowSize ({contextWindowSize}). " +
                                  $"Modelo: InputSize={model.InputSize}, Expected InputSize={tokenToIndex.Count * contextWindowSize}, OutputSize={model.OutputSize}.");
                throw new InvalidOperationException("Incompatibilidade entre o modelo e o vocabulário fixo/ContextWindowSize durante o treinamento.");
            }

            double chunkLoss = model.TrainEpoch(inputs, targets, learningRate);
            totalLoss += chunkLoss;
            Console.WriteLine($"Chunk {chunkIndex} processado, Perda: {chunkLoss:F4}");
        }

        private (Tensor[] inputs, Tensor[] targets) PrepareDataset(string chunkText)
        {
            var inputs = new List<Tensor>();
            var targets = new List<Tensor>();

            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            
            var matches = Regex.Matches(chunkText.ToLower(), pattern);
            var tokens = matches.Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToArray();

            // Adiciona padding no INÍCIO da sequência se o número de tokens for menor que a janela
            // Isso é para formar as primeiras sequências para o treinamento
            var paddedTokens = new List<string>();
            for (int k = 0; k < contextWindowSize; k++) // Pré-preenche com tokens de padding
            {
                paddedTokens.Add(padToken);
            }
            paddedTokens.AddRange(tokens); // Adiciona os tokens reais

            // Um par (inputs, targets) requer contextWindowSize tokens para a entrada e 1 token para o alvo.
            // Então, precisamos de pelo menos contextWindowSize + 1 tokens na lista `paddedTokens` para criar o primeiro par.
            // O loop vai até o penúltimo elemento, pois o último elemento será o alvo.
            for (int i = 0; i < paddedTokens.Count - contextWindowSize; i++)
            {
                // A sequência de entrada é a janela de tokens de 'contextWindowSize' antes do nextToken
                string[] currentWindowTokens = paddedTokens.Skip(i).Take(contextWindowSize).ToArray();
                string nextToken = paddedTokens[i + contextWindowSize]; // O token a ser previsto

                // Verifica se todos os tokens da janela e o próximo token estão no vocabulário
                if (!tokenToIndex.ContainsKey(nextToken) || !currentWindowTokens.All(t => tokenToIndex.ContainsKey(t)))
                {
                    Console.WriteLine($"Sequência ignorada no dataset (índice {i}): tokens ausentes no vocabulário fixo. Token de predição: '{nextToken}', Sequência de entrada: '{string.Join(" ", currentWindowTokens)}'");
                    continue;
                }

                // Cria o tensor de entrada como uma concatenação de one-hot vectors
                double[] inputData = new double[tokenToIndex.Count * contextWindowSize];
                for (int k = 0; k < contextWindowSize; k++)
                {
                    int tokenVocabIndex = tokenToIndex[currentWindowTokens[k]];
                    int offset = k * tokenToIndex.Count; // Offset para a posição do token na janela
                    inputData[offset + tokenVocabIndex] = 1.0;
                }
                inputs.Add(new Tensor(inputData, [tokenToIndex.Count * contextWindowSize]));

                // O alvo é o one-hot encoding do próximo token
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
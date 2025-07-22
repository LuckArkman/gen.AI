using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Core
{
    public class Trainer
    {
        private string datasetPath;
        private readonly string modelPathTemplate;
        private readonly string vocabPathTemplate;
        private NeuralNetwork model;
        private Dictionary<char, int> charToIndex;
        private List<char> indexToChar;
        private readonly int hiddenSize;
        private readonly int sequenceLength;
        private readonly double learningRate;
        private readonly int epochs;

        public Trainer(string datasetPath, string modelPathTemplate, string vocabPathTemplate,
            int hiddenSize = 256, int sequenceLength = 10, double learningRate = 0.01, int epochs = 10)
        {
            System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);

            if (string.IsNullOrEmpty(datasetPath))
                throw new ArgumentNullException(nameof(datasetPath));
            if (string.IsNullOrEmpty(modelPathTemplate))
                throw new ArgumentNullException(nameof(modelPathTemplate));
            if (string.IsNullOrEmpty(vocabPathTemplate))
                throw new ArgumentNullException(nameof(vocabPathTemplate));
            if (sequenceLength <= 0)
                throw new ArgumentException("SequenceLength deve ser positivo.", nameof(sequenceLength));
            if (learningRate <= 0)
                throw new ArgumentException("LearningRate deve ser positivo.", nameof(learningRate));
            if (epochs <= 0)
                throw new ArgumentException("Epochs deve ser positivo.", nameof(epochs));

            this.datasetPath = datasetPath;
            this.modelPathTemplate = modelPathTemplate;
            this.vocabPathTemplate = vocabPathTemplate;
            this.hiddenSize = hiddenSize;
            this.sequenceLength = sequenceLength;
            this.learningRate = learningRate;
            this.epochs = epochs;

            charToIndex = new Dictionary<char, int>();
            indexToChar = new List<char>();
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

                for (int epoch = startEpoch; epoch <= epochs; epoch++)
                {
                    Console.WriteLine($"Iniciando época {epoch}/{epochs}");

                    if (epoch > 1)
                    {
                        string prevModelPath = modelPathTemplate.Replace("{epoch}", (epoch - 1).ToString());
                        string prevVocabPath = vocabPathTemplate.Replace("{epoch}", (epoch - 1).ToString());
                        if (File.Exists(prevModelPath) && File.Exists(prevVocabPath))
                        {
                            Console.WriteLine($"Tentando carregar modelo e vocabulário da época {epoch - 1}...");
                            bool loaded = LoadModelAndVocabulary(prevModelPath, prevVocabPath);
                            if (!loaded)
                            {
                                Console.WriteLine($"Falha ao carregar modelo ou vocabulário da época {epoch - 1}. Reconstruindo vocabulário.");
                                BuildInitialVocabulary();
                                if (charToIndex.Count == 0)
                                {
                                    throw new InvalidOperationException("Nenhum caractere válido encontrado no dataset.");
                                }

                                model = new NeuralNetwork(charToIndex.Count, hiddenSize, charToIndex.Count);
                            }
                            else
                            {
                                // Verificar se o tamanho do vocabulário corresponde ao modelo
                                var modelData = JsonSerializer.Deserialize<NeuralNetworkModelData>(File.ReadAllText(prevModelPath));
                                if (modelData != null && modelData.InputSize != charToIndex.Count)
                                {
                                    Console.WriteLine($"Tamanho do vocabulário ({charToIndex.Count}) não corresponde ao modelo carregado ({modelData.InputSize}). Reconstruindo vocabulário.");
                                    BuildInitialVocabulary();
                                    if (charToIndex.Count == 0)
                                    {
                                        throw new InvalidOperationException("Nenhum caractere válido encontrado no dataset.");
                                    }

                                    model = new NeuralNetwork(charToIndex.Count, hiddenSize, charToIndex.Count);
                                }
                            }
                        }
                        else
                        {
                            Console.WriteLine($"Modelo ou vocabulário da época {epoch - 1} não encontrado. Reconstruindo vocabulário.");
                            BuildInitialVocabulary();
                            if (charToIndex.Count == 0)
                            {
                                throw new InvalidOperationException("Nenhum caractere válido encontrado no dataset.");
                            }

                            model = new NeuralNetwork(charToIndex.Count, hiddenSize, charToIndex.Count);
                        }
                    }
                    else
                    {
                        Console.WriteLine("Construindo vocabulário inicial...");
                        BuildInitialVocabulary();
                        if (charToIndex.Count == 0)
                        {
                            throw new InvalidOperationException("Nenhum caractere válido encontrado no dataset.");
                        }

                        model = new NeuralNetwork(charToIndex.Count, hiddenSize, charToIndex.Count);
                        Console.WriteLine(
                            $"Modelo inicializado com vocabulário de {charToIndex.Count} caracteres antes do processamento dos chunks.");
                    }

                    double totalLoss = 0;
                    int chunkCount = 0;

                    using (var reader = new StreamReader(datasetPath, Encoding.UTF8, true))
                    {
                        int lineNumber = 0;
                        while (!reader.EndOfStream)
                        {
                            lineNumber++;
                            string? line = reader.ReadLine();
                            if (string.IsNullOrEmpty(line)) continue;

                            chunkCount++;
                            //Console.WriteLine($"Processando linha {lineNumber} como chunk {chunkCount}...");
                            ProcessChunk(line, ref totalLoss, chunkCount, epoch == 1);
                            GC.Collect();
                        }
                    }

                    if (chunkCount == 0)
                    {
                        throw new InvalidOperationException("Nenhum chunk válido encontrado no dataset. Verifique o arquivo de entrada.");
                    }

                    double averageLoss = totalLoss / chunkCount;
                    Console.WriteLine($"Época {epoch}/{epochs} concluída. Perda média: {averageLoss:F4}, Total de chunks processados: {chunkCount}, Tamanho do vocabulário: {charToIndex.Count}");

                    string modelPath = modelPathTemplate.Replace("{epoch}", epoch.ToString());
                    string vocabPath = vocabPathTemplate.Replace("{epoch}", epoch.ToString());
                    try
                    {
                        model?.SaveModel(modelPath);
                        SaveVocabulary(vocabPath);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Aviso: Falha ao salvar modelo ou vocabulário para a época {epoch}: {ex.Message}");
                        Console.WriteLine("Continuando o treinamento sem salvar os arquivos.");
                    }
                }

                Console.WriteLine("Treinamento concluído.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro durante o treinamento: {ex.Message}");
                throw;
            }
        }

        private void ValidateFileEncoding()
        {
            try
            {
                string content = File.ReadAllText(datasetPath, Encoding.UTF8);
                if (content.Contains("\uFFFD"))
                {
                    throw new InvalidOperationException("O arquivo contém caracteres inválidos (substituição \uFFFD). Verifique a codificação do arquivo.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao validar a codificação do arquivo: {ex.Message}");
                throw;
            }
        }

        private void BuildInitialVocabulary()
        {
            try
            {
                charToIndex.Clear();
                indexToChar.Clear();
                using (var reader = new StreamReader(datasetPath, Encoding.UTF8, true))
                {
                    int lineNumber = 0;
                    int invalidCharCount = 0;
                    const int maxInvalidCharsToLog = 100; // Limite para evitar logs excessivos
                    while (!reader.EndOfStream)
                    {
                        lineNumber++;
                        string line = reader.ReadLine();
                        if (string.IsNullOrEmpty(line)) continue;
                        foreach (char c in line)
                        {
                            // Ignora caracteres de controle (exceto espaço), substituição (�) ou fora do intervalo Unicode válido
                            if (char.IsControl(c) && c != ' ' || c == '\uFFFD' || (int)c > 0x10FFFF)
                            {
                                if (invalidCharCount < maxInvalidCharsToLog)
                                {
                                    Console.WriteLine($"Caractere inválido ignorado na linha {lineNumber}: {(int)c} ({c})");
                                    invalidCharCount++;
                                }
                                else if (invalidCharCount == maxInvalidCharsToLog)
                                {
                                    Console.WriteLine($"Limite de caracteres inválidos atingido. Não serão registrados mais erros de caracteres na construção do vocabulário.");
                                    invalidCharCount++;
                                }

                                continue;
                            }

                            if (!charToIndex.ContainsKey(c))
                            {
                                charToIndex[c] = indexToChar.Count;
                                indexToChar.Add(c);
                                Console.WriteLine($"Novo caractere '{c}' (Unicode: {(int)c}) adicionado ao vocabulário inicial na linha {lineNumber}.");
                            }
                        }
                    }
                }

                if (charToIndex.Count == 0)
                {
                    throw new InvalidOperationException("Nenhum caractere válido encontrado no dataset.");
                }

                Console.WriteLine($"Vocabulário inicial construído. Tamanho: {charToIndex.Count} caracteres.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao construir vocabulário inicial: {ex.Message}");
                charToIndex.Clear();
                indexToChar.Clear();
                throw;
            }
        }

        public bool LoadModelAndVocabulary(string modelPath, string vocabPath)
        {
            try
            {
                model = NeuralNetwork.LoadModel(modelPath);
                if (model == null)
                {
                    Console.WriteLine($"Falha ao carregar o modelo de: {modelPath}");
                    return false;
                }

                LoadVocabulary(vocabPath);
                if (charToIndex.Count == 0)
                {
                    Console.WriteLine($"Falha ao carregar o vocabulário de: {vocabPath}");
                    return false;
                }

                Console.WriteLine($"Modelo e vocabulário carregados com sucesso de: {modelPath}, {vocabPath}");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar modelo ou vocabulário: {ex.Message}");
                return false;
            }
        }

        private void LoadVocabulary(string vocabPath)
        {
            try
            {
                charToIndex = new Dictionary<char, int>();
                indexToChar = new List<char>();
                using (var reader = new StreamReader(vocabPath, Encoding.UTF8, true))
                {
                    int lineNumber = 0;
                    int invalidCharCount = 0;
                    const int maxInvalidCharsToLog = 100;
                    while (!reader.EndOfStream)
                    {
                        lineNumber++;
                        string line = reader.ReadLine()?.Trim();
                        if (string.IsNullOrEmpty(line) || line.Length != 1)
                        {
                            Console.WriteLine($"Linha inválida ignorada no vocabulário na linha {lineNumber}: '{line}'");
                            continue;
                        }

                        char c = line[0];
                        if (char.IsControl(c) && c != ' ' || c == '\uFFFD' || (int)c > 0x10FFFF)
                        {
                            if (invalidCharCount < maxInvalidCharsToLog)
                            {
                                Console.WriteLine($"Caractere inválido ignorado no vocabulário na linha {lineNumber}: {(int)c} ({c})");
                                invalidCharCount++;
                            }
                            else if (invalidCharCount == maxInvalidCharsToLog)
                            {
                                Console.WriteLine($"Limite de caracteres inválidos atingido. Não serão registrados mais erros de caracteres no carregamento do vocabulário.");
                                invalidCharCount++;
                            }

                            continue;
                        }

                        if (!charToIndex.ContainsKey(c))
                        {
                            charToIndex[c] = indexToChar.Count;
                            indexToChar.Add(c);
                            Console.WriteLine($"Caractere '{c}' (Unicode: {(int)c}) carregado do vocabulário na linha {lineNumber}.");
                        }
                    }
                }

                if (indexToChar.Count == 0)
                {
                    throw new InvalidOperationException(
                        "Nenhum caractere válido encontrado no arquivo de vocabulário.");
                }

                Console.WriteLine($"Vocabulário carregado de: {vocabPath}, Tamanho: {charToIndex.Count} caracteres.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar vocabulário: {ex.Message}");
                charToIndex = new Dictionary<char, int>();
                indexToChar = new List<char>();
                throw;
            }
        }

        private void ProcessChunk(string chunkText, ref double totalLoss, int chunkIndex, bool buildVocabulary)
        {
            if (buildVocabulary)
            {
                int initialVocabSize = charToIndex.Count;
                int invalidCharCount = 0;
                const int maxInvalidCharsToLog = 100;
                foreach (char c in chunkText)
                {
                    if (char.IsControl(c) && c != ' ' || c == '\uFFFD' || (int)c > 0x10FFFF)
                    {
                        if (invalidCharCount < maxInvalidCharsToLog)
                        {
                            Console.WriteLine($"Caractere inválido ignorado no chunk {chunkIndex}: {(int)c} ({c})");
                            invalidCharCount++;
                        }
                        else if (invalidCharCount == maxInvalidCharsToLog)
                        {
                            Console.WriteLine($"Limite de caracteres inválidos atingido. Não serão registrados mais erros de caracteres no chunk {chunkIndex}.");
                            invalidCharCount++;
                        }

                        continue;
                    }

                    if (!charToIndex.ContainsKey(c))
                    {
                        charToIndex[c] = indexToChar.Count;
                        indexToChar.Add(c);
                        Console.WriteLine($"Novo caractere '{c}' (Unicode: {(int)c}) adicionado ao vocabulário no chunk {chunkIndex}.");
                    }
                }

                if (charToIndex.Count > initialVocabSize)
                {
                    Console.WriteLine($"Vocabulário atualizado no chunk {chunkIndex}. Novo tamanho: {charToIndex.Count} caracteres.");
                }
            }

            var (inputs, targets) = PrepareDataset(chunkText);
            if (inputs.Length == 0 || targets.Length == 0)
            {
                Console.WriteLine($"Chunk {chunkIndex} ignorado: nenhum dado válido para treinamento.");
                return;
            }

            if (model == null)
            {
                throw new InvalidOperationException($"Modelo não inicializado para o chunk {chunkIndex}. Verifique o vocabulário e os dados de entrada.");
            }

            double chunkLoss = model.TrainEpoch(inputs, targets, learningRate);
            totalLoss += chunkLoss;
            Console.WriteLine($"Chunk {chunkIndex} processado, Perda: {chunkLoss:F4}");
        }

        private (Tensor[] inputs, Tensor[] targets) PrepareDataset(string text)
        {
            var inputs = new List<Tensor>();
            var targets = new List<Tensor>();

            var cleanText = text.Replace("\n", " ").Trim();
            if (cleanText.Length < sequenceLength + 1)
            {
                Console.WriteLine($"Linha ignorada: comprimento insuficiente ({cleanText.Length} < {sequenceLength + 1}).");
                return (inputs.ToArray(), targets.ToArray());
            }

            for (int i = 0; i < cleanText.Length - sequenceLength; i++)
            {
                string sequence = cleanText.Substring(i, sequenceLength);
                char nextChar = cleanText[i + sequenceLength];

                if (!charToIndex.ContainsKey(nextChar) || !sequence.All(c => charToIndex.ContainsKey(c))) continue;

                double[] inputData = new double[charToIndex.Count];
                inputData[charToIndex[sequence[sequenceLength - 1]]] = 1.0;
                inputs.Add(new Tensor(inputData, new int[] { charToIndex.Count }));

                double[] targetData = new double[charToIndex.Count];
                targetData[charToIndex[nextChar]] = 1.0;
                targets.Add(new Tensor(targetData, new int[] { charToIndex.Count }));
            }

            //Console.WriteLine($"PrepareDataset retornou {inputs.Count} entradas e {targets.Count} alvos.");
            return (inputs.ToArray(), targets.ToArray());
        }

        private void SaveVocabulary(string vocabPath)
        {
            try
            {
                using (var writer = new StreamWriter(vocabPath, false, Encoding.UTF8))
                {
                    foreach (char c in indexToChar)
                    {
                        writer.WriteLine(c);
                    }
                }

                Console.WriteLine($"Vocabulário salvo em: {vocabPath}, Tamanho: {charToIndex.Count} caracteres.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar vocabulário: {ex.Message}");
            }
        }
    }
}
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using BinaryTreeSwapFile;
using Models;
using Services;

namespace Core
{
    public class Trainer
    {
        private readonly string datasetPath;
        private readonly string modelPathTemplate;
        private readonly string _vocabPath;
        private NeuralNetwork? model;
        private Dictionary<string, int> tokenToIndex = new Dictionary<string, int>();
        private List<string> indexToToken = new List<string>();
        private readonly int hiddenSize;
        private readonly int contextWindowSize;
        private double learningRate; // MUDANÇA: Agora é um campo para poder ser modificado
        private readonly int epochs;
        private readonly string padToken = "[PAD]";
        private readonly string logPath;
        private readonly TextProcessorService _textProcessorService;
        private readonly BinaryTreeFileStorage _memoryStorage;
        private readonly DatasetService _datasetService;

        // --- PARÂMETROS PARA AJUSTE DA TAXA DE APRENDIZADO ---
        private readonly List<double> _lossHistory = new List<double>();
        private readonly int _lrDecisionWindow = 5; // Janela decisiva: 5 épocas para observar
        private int _epochsWithoutImprovement = 0;
        private readonly double _lrReductionFactor = 0.5; // Reduz a LR pela metade
        private readonly double _minLearningRate = 1e-6; // Limite mínimo para a LR
        private double _bestLoss = double.MaxValue;

        public Trainer(string datasetPath,
            string modelPathTemplate,
            string vocabPath,
            int hiddenSize,
            int sequenceLength,
            double initialLearningRate,
            int epochs,
            TextProcessorService textProcessorService,
            BinaryTreeFileStorage memoryStorage,
            DatasetService datasetService)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            if (string.IsNullOrEmpty(datasetPath)) throw new ArgumentNullException(nameof(datasetPath));
            if (string.IsNullOrEmpty(modelPathTemplate)) throw new ArgumentNullException(nameof(modelPathTemplate));
            // ... (outras validações)

            this.datasetPath = datasetPath;
            this.modelPathTemplate = modelPathTemplate;
            this._vocabPath = vocabPath;
            this.hiddenSize = hiddenSize;
            this.contextWindowSize = sequenceLength;
            this.learningRate = initialLearningRate; // Atribui ao campo
            this.epochs = epochs;
            _datasetService = datasetService;
            this.logPath = Path.Combine(Path.GetDirectoryName(datasetPath) ?? "", "training_log.txt");
            _textProcessorService = textProcessorService;
            _memoryStorage = memoryStorage;
        }

        public void Train(int startEpoch = 1, int batchSize = 32)
        {
            try
            {
                SetupVocabularyAndModel();
                if (model == null) throw new InvalidOperationException("Falha ao inicializar o modelo.");

                // --- FASE DE PRÉ-PROCESSAMENTO COM STREAMING E MONITORAMENTO DE CHUNKS ---
                Console.WriteLine("Iniciando pré-processamento do dataset via streaming...");
                _memoryStorage.Clear();

                // 1. CALCULA O TOTAL DE CHUNKS PARA O MONITORAMENTO
                Console.WriteLine("Analisando o tamanho do dataset para monitorar o progresso...");
                long totalLines = File.ReadLines(datasetPath).LongCount();
                const int linesPerSuperChunk = 10000;
                // Evita divisão por zero se o arquivo estiver vazio
                int totalChunks = (totalLines > 0) ? (int)Math.Ceiling((double)totalLines / linesPerSuperChunk) : 0;
                Console.WriteLine(
                    $"Dataset contém {totalLines} linhas, que serão processadas em {totalChunks} blocos.");

                var allSampleOffsets = new List<long>();
                int superChunkCount = 0;

                if (totalChunks > 0)
                {
                    using (var reader = new StreamReader(datasetPath, Encoding.UTF8))
                    {
                        bool endOfFile = false;
                        while (!endOfFile)
                        {
                            superChunkCount++;
                            Console.WriteLine(
                                $"\nLendo e processando o bloco de texto nº {superChunkCount}/{totalChunks}...");

                            var lines = new List<string>(linesPerSuperChunk);
                            for (int i = 0; i < linesPerSuperChunk; i++)
                            {
                                string? line = reader.ReadLine();
                                if (line == null)
                                {
                                    endOfFile = true;
                                    break;
                                }

                                lines.Add(line);
                            }

                            if (lines.Count > 0)
                            {
                                string textBlock = string.Join("\n", lines);
                                var offsetsForBlock = _datasetService.PreprocessAndStoreSamples(
                                    textBlock, contextWindowSize, tokenToIndex, padToken, _memoryStorage);
                                allSampleOffsets.AddRange(offsetsForBlock);

                                // 2. EMITE A MENSAGEM DE PROGRESSO APÓS CADA CHUNK SALVO
                                double percentage = (double)superChunkCount / totalChunks * 100;
                                Console.ForegroundColor = ConsoleColor.Green;
                                Console.WriteLine(
                                    $"[Progresso] Bloco {superChunkCount} de {totalChunks} armazenado com sucesso ({percentage:F1}% concluído).");
                                Console.ResetColor();
                            }
                        }
                    }
                }

                Console.WriteLine(
                    $"\nPré-processamento via streaming concluído. Total de {allSampleOffsets.Count} amostras armazenadas.");

                if (allSampleOffsets.Count == 0)
                {
                    throw new InvalidOperationException(
                        "Nenhuma amostra de treinamento pôde ser gerada a partir do dataset.");
                }

                // --- FASE DE TREINAMENTO (O resto do método permanece o mesmo) ---
                for (int epoch = startEpoch; epoch <= epochs; epoch++)
                {
                    Console.WriteLine($"\nIniciando época {epoch}/{epochs} com Taxa de Aprendizado: {learningRate:F6}");
                    double epochTotalLoss = 0;
                    long totalSamplesInEpoch = 0;

                    var random = new Random();
                    var shuffledOffsets = allSampleOffsets.OrderBy(x => random.Next()).ToList();

                    int totalBatches = (int)Math.Ceiling((double)shuffledOffsets.Count / batchSize);
                    for (int i = 0; i < shuffledOffsets.Count; i += batchSize)
                    {
                        var offsetBatch = shuffledOffsets.Skip(i).Take(batchSize).ToList();
                        var miniBatch = new List<(Tensor input, Tensor target)>(offsetBatch.Count);

                        foreach (long offset in offsetBatch)
                        {
                            miniBatch.Add(_datasetService.GetSampleFromStorage(offset, _memoryStorage));
                        }

                        if (miniBatch.Count > 0)
                        {
                            double batchLoss = model.TrainEpoch(miniBatch, learningRate, epoch);
                            epochTotalLoss += batchLoss * miniBatch.Count;
                            totalSamplesInEpoch += miniBatch.Count;
                        }

                        Console.Write(
                            $"\rÉpoca {epoch}/{epochs}, Batch {(i / batchSize) + 1}/{totalBatches} processado...");
                    }

                    Console.WriteLine();
                    if (totalSamplesInEpoch == 0) continue;

                    double averageLoss = epochTotalLoss / totalSamplesInEpoch;
                    _lossHistory.Add(averageLoss);

                    var logMessage = $"Época {epoch}/{epochs} concluída. Perda média: {averageLoss:F4}";
                    Console.WriteLine(logMessage);
                    File.AppendAllText(logPath, logMessage + Environment.NewLine);

                    AdjustLearningRate(averageLoss);
                    model.SaveModel(modelPathTemplate);
                }

                Console.WriteLine("Treinamento concluído.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro crítico durante o treinamento: {ex.Message}");
                throw;
            }
        }

        private void SetupVocabularyAndModel()
        {
            if (File.Exists(_vocabPath))
            {
                Console.WriteLine($"Tentando carregar vocabulário existente de: {_vocabPath}");
                LoadVocabulary(_vocabPath);
            }
            else
            {
                Console.WriteLine("Nenhum vocabulário encontrado. Construindo vocabulário do dataset.");
                BuildVocabularyFromDataset();
            }

            if (tokenToIndex.Count <= 1)
            {
                throw new InvalidOperationException("Nenhum token válido encontrado para construir o vocabulário.");
            }

            if (File.Exists(modelPathTemplate))
            {
                Console.WriteLine($"Tentando carregar modelo de: {modelPathTemplate}...");
                model = NeuralNetwork.LoadModel(modelPathTemplate);
            }

            if (model == null || model.InputSize != tokenToIndex.Count * contextWindowSize ||
                model.OutputSize != tokenToIndex.Count)
            {
                if (model != null)
                {
                    Console.WriteLine($"Modelo existente é incompatível com o vocabulário atual. Criando novo modelo.");
                }
                else
                {
                    Console.WriteLine(
                        "Nenhum modelo anterior encontrado ou iniciando do zero. Inicializando novo modelo.");
                }

                model = new NeuralNetwork(tokenToIndex.Count * contextWindowSize, hiddenSize, tokenToIndex.Count,
                    contextWindowSize, learningRate);
            }
            else
            {
                Console.WriteLine($"Modelo Previo carregado com sucesso.");
            }
        }

        private void AdjustLearningRate(double currentLoss)
        {
            double min_delta = 0.0001; // Considera melhora apenas se for maior que isso
            if (currentLoss < _bestLoss - min_delta)
            {
                _bestLoss = currentLoss;
                _epochsWithoutImprovement = 0;
                Console.WriteLine($"Nova melhor perda encontrada: {_bestLoss:F4}. Contador de paciência zerado.");
            }
            else
            {
                _epochsWithoutImprovement++;
                Console.WriteLine(
                    $"Nenhuma melhoria significativa na perda. Épocas sem melhoria: {_epochsWithoutImprovement}/{_lrDecisionWindow}");
            }

            if (_epochsWithoutImprovement >= _lrDecisionWindow)
            {
                if (learningRate > _minLearningRate)
                {
                    double oldLr = learningRate;
                    learningRate *= _lrReductionFactor;
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine(
                        $"Platô de treinamento detectado! Reduzindo a taxa de aprendizado de {oldLr:F6} para {learningRate:F6}.");
                    Console.ResetColor();
                    _epochsWithoutImprovement = 0; // Reseta o contador para dar tempo ao modelo de se ajustar
                }
                else
                {
                    Console.WriteLine("Platô detectado, mas a taxa de aprendizado já está no seu valor mínimo.");
                }
            }
        }

        private void LoadVocabulary(string vocabPath)
        {
            tokenToIndex = new Dictionary<string, int>();
            indexToToken = new List<string>();
            using (var reader = new StreamReader(vocabPath, Encoding.UTF8))
            {
                string? line;
                while ((line = reader.ReadLine()) != null)
                {
                    string token = line.Trim();
                    if (!string.IsNullOrEmpty(token) && !tokenToIndex.ContainsKey(token))
                    {
                        tokenToIndex[token] = indexToToken.Count;
                        indexToToken.Add(token);
                    }
                }
            }

            Console.WriteLine($"Vocabulário carregado de: {vocabPath}, Tamanho: {tokenToIndex.Count} tokens.");
        }

        private void BuildVocabularyFromDataset()
        {
            tokenToIndex.Clear();
            indexToToken.Clear();
            tokenToIndex[padToken] = indexToToken.Count;
            indexToToken.Add(padToken);

            var specialChars = new[]
            {
                '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
                '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'
            };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";

            var uniqueTokens = new HashSet<string>();
            string fullText = File.ReadAllText(datasetPath, Encoding.UTF8);
            var matches = Regex.Matches(fullText.ToLower(), pattern);
            foreach (Match match in matches)
            {
                uniqueTokens.Add(match.Value);
            }

            foreach (string token in uniqueTokens.OrderBy(t => t))
            {
                if (!tokenToIndex.ContainsKey(token))
                {
                    tokenToIndex[token] = indexToToken.Count;
                    indexToToken.Add(token);
                }
            }

            Console.WriteLine($"Vocabulário inicial construído. Tamanho: {tokenToIndex.Count} tokens.");
            SaveVocabulary(_vocabPath);
        }

        private void SaveVocabulary(string vocabPath)
        {
            try
            {
                File.WriteAllLines(vocabPath, indexToToken, new UTF8Encoding(false));
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
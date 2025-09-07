using System.Text;
using System.Text.RegularExpressions;
using BinaryTreeSwapFile;
using GenerativeAIAPI.Controllers;
using Models;
using Services;

namespace Core;

public class ListenerService : IHostedService
{
    private readonly string modelDir;
    private readonly string vocabPath;
    private readonly string modelPath;
    private BinaryTreeFileStorage _memoryStorage;
    private readonly TextProcessorService _textProcessorService;
    private readonly string _memoryFilePath;
    private Dictionary<string, int> tokenToIndex;
    private readonly string padToken = "[PAD]";
    private readonly int contextWindowSize;
    private List<string> indexToToken;
    private NeuralNetwork? model;
    private readonly GenerativeAIController _server;
    private readonly ILogger<ListenerService> _logger;
    private const double KnowledgeInternalizationLearningRate = 0.001;

    public ListenerService(IConfiguration configuration,
        GenerativeAIController server,
        TextProcessorService textProcessorService,
        ILogger<ListenerService> logger)
    {
        _textProcessorService = textProcessorService;
        _server = server;
        _logger = logger;
        _server.ContextAdded += OnContextAdded;
        modelDir = configuration["ModelSettings:ModelDirectory"] ??
                   "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/";
        _memoryFilePath = configuration["ModelSettings:MemoryFilePath"] ?? Path.Combine(modelDir, "AIModelMem.dat");
        modelPath = Path.Combine(modelDir, $"model.json");
        vocabPath = Path.Combine(modelDir, "vocab.txt");
        model = NeuralNetwork.LoadModel(modelPath);
        _memoryStorage = new BinaryTreeFileStorage(_memoryFilePath);
        tokenToIndex = new Dictionary<string, int>();
        if (System.IO.File.Exists(modelPath) && System.IO.File.Exists(vocabPath))
        {
            try
            {
                LoadVocabulary();
                if (tokenToIndex.Count > 0)
                {
                    model = NeuralNetwork.LoadModel(modelPath);
                    if (model != null &&
                        (model.InputSize != tokenToIndex.Count * contextWindowSize ||
                         model.OutputSize != tokenToIndex.Count))
                    {
                        Console.WriteLine(
                            $"Modelo carregado, mas suas dimensões ({model.InputSize}, {model.OutputSize}) não correspondem ao tamanho do vocabulário ({tokenToIndex.Count}) e ContextWindowSize ({contextWindowSize}). O modelo pode ser incompatível.");
                        model = null;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao inicializar o controlador: {ex.Message}");
                model = null;
                tokenToIndex.Clear();
                indexToToken.Clear();
            }
        }
        else
        {
            Console.WriteLine(
                "Modelo ou vocabulário não encontrados na inicialização do controlador. Treine o modelo primeiro.");
        }

        if (!System.IO.File.Exists(_memoryFilePath) ||
            new FileInfo(_memoryFilePath).Length < sizeof(long) + TreeNode.NodeSize)
        {
            Console.WriteLine("Arquivo de memória virtual não encontrado ou vazio. Gerando árvore vazia...");
            _memoryStorage.GenerateEmptyTree();
        }
    }

    private void LoadVocabulary()
    {
        try
        {
            tokenToIndex = new Dictionary<string, int>();
            indexToToken = new List<string>();
            tokenToIndex[padToken] = indexToToken.Count;
            indexToToken.Add(padToken);

            using (var reader = new StreamReader(vocabPath, Encoding.UTF8, true))
            {
                int lineNumber = 0;
                while (!reader.EndOfStream)
                {
                    lineNumber++;
                    string line = reader.ReadLine()?.Trim();
                    if (string.IsNullOrEmpty(line))
                    {
                        Console.WriteLine(
                            $"Linha inválida ignorada no vocabulário na linha {lineNumber}: '{line}'");
                        continue;
                    }

                    string token = line;
                    if (token == padToken && tokenToIndex.ContainsKey(padToken)) continue;

                    if (char.IsControl(token[0]) && token[0] != ' ' || token == "\uFFFD" ||
                        (int)token[0] > 0x10FFFF)
                    {
                        Console.WriteLine($"Token inválido ignorado no vocabulário na linha {lineNumber}: {token}");
                        continue;
                    }

                    if (!tokenToIndex.ContainsKey(token))
                    {
                        tokenToIndex[token] = indexToToken.Count;
                        indexToToken.Add(token);
                    }
                }
            }

            if (indexToToken.Count == 0)
            {
                throw new InvalidOperationException("Nenhum token válido encontrado no arquivo de vocabulário.");
            }

            Console.WriteLine($"Vocabulário carregado de: {vocabPath}, Tamanho: {tokenToIndex.Count} tokens.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro ao carregar vocabulário: {ex.Message}");
            tokenToIndex = new Dictionary<string, int>();
            indexToToken = new List<string>();
            throw;
        }
    }

    async void OnContextAdded(object? sender, SaveContext contextManager)
    {
        long offsetToUpdateOrInsert = contextManager.offsetToUpdateOrInsert;
        byte[] serializedData = contextManager.serializedData;
        string newSummary = contextManager.newSummary;
        if (offsetToUpdateOrInsert != -1)
        {
            _memoryStorage.UpdateData(offsetToUpdateOrInsert, Encoding.UTF8.GetString(serializedData));
        }
        else
        {
            _memoryStorage.Insert(Encoding.UTF8.GetString(serializedData));
        }

        var loss = InternalizeKnowledgeIntoModel(newSummary);
        Console.WriteLine($"Perda na internalização de conhecimento: {loss:F4}");
        _memoryStorage.CleanUnusedNodes(TimeSpan.FromDays(30));
    }

    double InternalizeKnowledgeIntoModel(string knowledgeText)
    {
        if (model == null || tokenToIndex.Count == 0)
        {
            Console.WriteLine("Modelo ou vocabulário não inicializados para internalizar conhecimento.");
            return 0;
        }

        var dataset = PrepareDataset(knowledgeText, contextWindowSize);
        if (dataset.Count == 0)
        {
            Console.WriteLine("Dados de conhecimento insuficientes para internalização.");
            return 0;
        }

        Console.WriteLine($"Internalizando {dataset.Count} sequências de conhecimento no modelo...");
        double loss = model.TrainEpoch(dataset, KnowledgeInternalizationLearningRate);
        Console.WriteLine($"Perda na internalização de conhecimento: {loss:F4}");
        model.SaveModel(modelPath);
        return loss;
    }

    public List<(Tensor input, Tensor target)> PrepareDataset(string text, int currentContextWindowSize)
    {
        var dataset = new List<(Tensor input, Tensor target)>();

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
                Console.WriteLine(
                    $"Sequência ignorada no dataset (índice {i}): tokens ausentes no vocabulário. Próximo Token: '{nextToken}', Janela: '{string.Join(" ", currentWindowTokens)}'");
                continue;
            }

            double[] inputData = new double[tokenToIndex.Count * currentContextWindowSize];
            for (int k = 0; k < currentContextWindowSize; k++)
            {
                int tokenVocabIndex = tokenToIndex[currentWindowTokens[k]];
                int offset = k * tokenToIndex.Count;
                inputData[offset + tokenVocabIndex] = 1.0;
            }
            var inputTensor = new Tensor(inputData, new int[] { tokenToIndex.Count * currentContextWindowSize });

            double[] targetData = new double[tokenToIndex.Count];
            targetData[tokenToIndex[nextToken]] = 1.0;
            var targetTensor = new Tensor(targetData, new int[] { tokenToIndex.Count });
            
            dataset.Add((inputTensor, targetTensor));
        }

        return dataset;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _server.ContextAdded -= OnContextAdded;
        await Task.CompletedTask;
    }
}
using System.Text;
using System.Text.RegularExpressions;
using BinaryTreeSwapFile;
using GenerativeAIAPI.Controllers;
using Models;
using Services;

namespace Core;

public class ListenerService
{
    private readonly string modelDir;
    private readonly string vocabPath;
    private readonly string modelPath;
    private BinaryTreeFileStorage _memoryStorage;
    private readonly TextProcessorService _textProcessorService;
    private readonly DatasetService _datasetService;
    private readonly string _memoryFilePath;
    private Dictionary<string, int> tokenToIndex;
    private readonly string padToken = "[PAD]";
    private readonly int contextWindowSize;
    private List<string> indexToToken;
    private NeuralNetwork? model;
    private readonly ILogger<ListenerService> _logger;
    private const double KnowledgeInternalizationLearningRate = 0.001;

    public ListenerService(IConfiguration configuration,
        TextProcessorService textProcessorService,
        ILogger<ListenerService> logger,
        DatasetService datasetService)
    {
        _textProcessorService = textProcessorService;
        _logger = logger;
        _datasetService = datasetService;
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

    public async Task OnContextAdded(object? sender, SaveContext contextManager)
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

        var dataset = _datasetService.PrepareDataset(knowledgeText, contextWindowSize, tokenToIndex, padToken);
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
}
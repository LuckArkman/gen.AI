using Microsoft.AspNetCore.Mvc;
using System.Text;
using System.Text.RegularExpressions;
using Core;
using Models;
using Microsoft.Extensions.Configuration;
using System.IO;
using System.Collections.Generic;
using System;
using System.Linq;
using System.Text.Json;
using BinaryTreeSwapFile;
using Services;

namespace GenerativeAIAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class GenerativeAIController : ControllerBase
    {
        private readonly ListenerService _listenerService;
        private readonly DatasetService _datasetService;
        private readonly string modelDir;
        private readonly string modelPath;
        private readonly string vocabPath;
        private NeuralNetwork? model;
        private Dictionary<string, int> tokenToIndex;
        private List<string> indexToToken;
        private const int HiddenSize = 512;
        private readonly string padToken = "[PAD]";
        private readonly int contextWindowSize;
        private BinaryTreeSwapFile.BinaryTreeFileStorage _memoryStorage;
        private readonly TextProcessorService _textProcessorService;
        private readonly string _memoryFilePath;
        private readonly ContextManager _contextManager;
        private readonly KnowledgeAcquisitionService _knowledgeAcquisitionService;
        private readonly ILogger<ListenerService> _logger;
        Queue<SaveContext> _ContextAdded = new();
        public event EventHandler<SaveContext>? ContextAdded;

        private const double KnowledgeInternalizationLearningRate = 0.01;

        public GenerativeAIController(IConfiguration configuration,
            ContextManager contextManager,
            TextProcessorService textProcessorService,
            KnowledgeAcquisitionService knowledgeAcquisitionService,
            ListenerService listenerService,
            ILogger<ListenerService> logger,
            DatasetService datasetService) // ListenerService injetado
        {
            modelDir = configuration["ModelSettings:ModelDirectory"] ??
                       "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/"; // CORRIGIDO: Inicializa modelDir primeiro

            _contextManager = contextManager;
            _textProcessorService = textProcessorService;
            _knowledgeAcquisitionService = knowledgeAcquisitionService;
            _listenerService = listenerService; // Atribui o serviço
            _logger = logger;
            _datasetService = datasetService;

            _memoryFilePath = configuration["ModelSettings:MemoryFilePath"] ?? Path.Combine(modelDir, "AIModelMem.dat");
            _memoryStorage = new BinaryTreeSwapFile.BinaryTreeFileStorage(_memoryFilePath);
            if (!System.IO.File.Exists(_memoryFilePath) ||
                new FileInfo(_memoryFilePath).Length < sizeof(long) + TreeNode.NodeSize)
            {
                Console.WriteLine("Arquivo de memória virtual não encontrado ou vazio. Gerando árvore vazia...");
                _memoryStorage.GenerateEmptyTree();
            }

            if (!Directory.Exists(modelDir))
            {
                Console.WriteLine($"Aviso: O diretório do modelo '{modelDir}' não existe na inicialização da API.");
            }

            contextWindowSize = configuration.GetValue<int>("ModelSettings:ContextWindowSize", 10);

            modelPath = Path.Combine(modelDir, $"model.json");
            vocabPath = Path.Combine(modelDir, "vocab.txt");

            tokenToIndex = new Dictionary<string, int>();
            indexToToken = new List<string>();

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
        }

        private bool IsValidText(string? text)
        {
            if (string.IsNullOrEmpty(text)) return true;

            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";

            var matches = Regex.Matches(text.ToLower(), pattern);
            return matches.All(m => tokenToIndex.ContainsKey(m.Value));
        }

        [HttpPost("train")]
        public IActionResult Train([FromBody] TrainRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.TextData)) return BadRequest(new { Error = "TextData não pode estar vazio." });
                if (request.ContextWindowSize <= 0) return BadRequest(new { Error = "ContextWindowSize deve ser positivo." });

                if (System.IO.File.Exists(vocabPath)) LoadVocabulary();
                if (tokenToIndex.Count <= 1) BuildVocabulary(request.TextData);
                
                int vocabSize = tokenToIndex.Count;
                if (model == null || model.InputSize != vocabSize * request.ContextWindowSize || model.OutputSize != vocabSize)
                {
                    model = new NeuralNetwork(vocabSize * request.ContextWindowSize, HiddenSize, vocabSize, request.ContextWindowSize);
                }

                // CORREÇÃO: Usar o método correto para criar os batches para o treinamento via API
                // O chunkSize do request é o nosso batchSize
                int batchSize = request.ContextWindowSize; // Assumindo que chunkSize é passado aqui
                var batches = _datasetService.PrepareBatchesFromText(request.TextData, request.ContextWindowSize, tokenToIndex, padToken, batchSize);

                if (batches.Count == 0)
                {
                    return BadRequest(new { Error = "Dados de treinamento insuficientes para a ContextWindowSize especificada." });
                }

                double learningRate = request.LearningRate ?? 0.001;
                int epochs = request.Epochs ?? 10;
                var losses = new List<double>();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double totalLossForEpoch = 0;
                    long totalSamplesInEpoch = 0;

                    // Itera sobre cada batch gerado a partir do texto
                    foreach (var miniBatch in batches)
                    {
                        if(miniBatch.Count > 0)
                        {
                            double batchLoss = model.TrainEpoch(miniBatch, learningRate, epoch);
                            totalLossForEpoch += batchLoss * miniBatch.Count;
                            totalSamplesInEpoch += miniBatch.Count;
                        }
                    }
                    
                    if (totalSamplesInEpoch > 0)
                    {
                        double averageLoss = totalLossForEpoch / totalSamplesInEpoch;
                        losses.Add(averageLoss);
                        Console.WriteLine($"Época {epoch + 1}/{epochs}, Perda: {averageLoss:F4}");
                    }
                }

                model.SaveModel(modelPath);
                SaveVocabulary();

                return Ok(new
                {
                    Message = "Treinamento concluído", 
                    AverageLoss = losses.Any() ? losses.Average() : 0, 
                    VocabularySize = vocabSize,
                    EpochLosses = losses
                });
            }
            catch (Exception ex)
            {
                return BadRequest(new { Error = $"Falha no treinamento: {ex.Message}" });
            }
        }

        [HttpPost("test")]
        public IActionResult Test([FromBody] TestRequest request)
        {
            try
            {
                if (model == null || tokenToIndex.Count == 0)
                {
                    return BadRequest(new { Error = "Modelo ou vocabulário não inicializados. Treine o modelo primeiro." });
                }
                
                // CORREÇÃO: O endpoint de teste deve avaliar a perda em amostras individuais,
                // então usamos um método que retorna uma lista simples de amostras.
                // Usaremos PrepareSingleBatch, que retorna List<(Tensor, Tensor)>
                var dataset = _datasetService.PrepareSingleBatch(request.TextData!, request.ContextWindowSize, tokenToIndex, padToken);
                
                if (dataset.Count == 0)
                {
                    return BadRequest(new { Error = "Dados de teste insuficientes para a ContextWindowSize especificada." });
                }

                double totalLoss = 0;
                foreach (var (input, target) in dataset)
                {
                    Tensor output = model.Forward(input);
                    for (int o = 0; o < tokenToIndex.Count; o++)
                    {
                        if (target.Infer(new int[] { o }) == 1.0)
                        {
                            double outputValue = output.Infer(new int[] { o });
                            totalLoss += -Math.Log(Math.Max(outputValue, 1e-9));
                            break;
                        }
                    }
                }

                double averageLoss = dataset.Count > 0 ? totalLoss / dataset.Count : 0;
                return Ok(new { Message = "Teste concluído", AverageLoss = averageLoss });
            }
            catch (Exception ex)
            {
                return BadRequest(new { Error = $"Falha no teste: {ex.Message}" });
            }
        }

        [HttpPost("generate")]
        public async Task<IActionResult> Generate([FromBody] GenerateRequest request)
        {
            try
            {
                if (model == null || tokenToIndex.Count == 0) return BadRequest(new { Error = "Modelo não inicializado." });
                if (string.IsNullOrEmpty(request.SeedText)) return BadRequest(new { Error = "SeedText não pode estar vazio." });

                string enrichedContext = string.Empty;
                string knowledgeSource = "Conhecimento Generativo Internalizado";

                // --- CORREÇÃO: APLICAR A LÓGICA DE DETECÇÃO DE INTENÇÃO ---
                bool isInformationalQuery = IsInformationalQuery(request.SeedText);

                if (isInformationalQuery)
                {
                    Console.WriteLine("Intenção de busca de informação detectada.");
                    // Executa o fluxo de aquisição de conhecimento apenas para perguntas informacionais
                    (enrichedContext, knowledgeSource) = await AcquireAndEnrichKnowledge(request.SeedText);
                }
                else
                {
                    Console.WriteLine("Intenção social/conversacional detectada. Pulando busca externa.");
                }

                // --- 2. CONSTRUÇÃO DO PROMPT E GERAÇÃO ---
                string effectiveSeed;
                if (isInformationalQuery)
                {
                    effectiveSeed = $"Pergunta do usuário: '{request.SeedText}'. Com base nos fatos conhecidos: '{enrichedContext}', elabore uma resposta. Resposta:";
                }
                else
                {
                    effectiveSeed = request.SeedText; // Para saudações, usa o texto original
                }

                Console.WriteLine($"--- PROMPT PARA O MODELO ---\n{effectiveSeed}\n--------------------------");

                // --- 3. GERAÇÃO DE TEXTO PELO MODELO LSTM ---
                string finalGeneratedText = GenerateText(effectiveSeed, request.Length ?? 50, request.Temperature);
                
                await _contextManager.StoreConversationContext(request.SeedText, finalGeneratedText);
                
                Console.WriteLine($"Fonte utilizada: {knowledgeSource}");
                Console.WriteLine($"AI response: {finalGeneratedText}");

                return Ok(new { response = finalGeneratedText, source = knowledgeSource });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { Error = $"Falha na geração: {ex.Message}" });
            }
        }
        
        private async Task<(string enrichedContext, string knowledgeSource)> AcquireAndEnrichKnowledge(string query)
        {
            string topic = _textProcessorService.ExtractMainTopic(query);
            string newSummary = string.Empty;
            string memorySummary = string.Empty;
            string source = "N/A";

            ContextInfo? storedContext = _contextManager.GetContextByTopic(topic);
            if (storedContext != null) memorySummary = storedContext.Summary;

            bool needsExternalAcquisition = storedContext == null || (DateTime.UtcNow - new DateTime(storedContext.ExternalLastUpdatedTicks) > _contextManager.MaxContextAgeForRefresh);

            if (needsExternalAcquisition)
            {
                var (externalContent, sourceName) = await _knowledgeAcquisitionService.GetInformation(topic);
                if (externalContent.Any() && !string.IsNullOrWhiteSpace(string.Join("", externalContent)))
                {
                    source = $"Fonte principal: {sourceName}";
                    string combinedExternalContent = string.Join(" ", externalContent);
                    ExpandVocabularyAndAdaptModel(combinedExternalContent); // AVISO: Esta função é problemática
                    newSummary = _textProcessorService.Summarize(combinedExternalContent);

                    // ... (Lógica de salvar contexto e disparar evento) ...
                }
            }

            if (!string.IsNullOrEmpty(newSummary)) return (newSummary, source);
            if (!string.IsNullOrEmpty(memorySummary)) return (memorySummary, "Fonte principal: Memória Virtual");
            
            return (string.Empty, "Nenhum conhecimento encontrado.");
        }
        
        private bool IsInformationalQuery(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) return false;
            var lowerText = text.ToLower().Trim();
            var socialPhrases = new HashSet<string>
            {
                "olá", "oi", "bom dia", "boa tarde", "boa noite", "tudo bem?", "tudo bom?",
                "como vai?", "e aí?", "obrigado", "obrigada", "de nada", "valeu", "tchau", "até mais"
            };
            if (socialPhrases.Contains(lowerText.Trim('?', '!', '.'))) return false;

            var informationalKeywords = new[] { "o que é", "quem foi", "explique", "fale sobre", "qual é", "como funciona" };
            if (informationalKeywords.Any(keyword => lowerText.Contains(keyword))) return true;

            if (lowerText.Split(' ').Length > 4) return true;
            
            return false;
        }
        
        private StringBuilder generatedTextBuilder = new StringBuilder();
        private Random rand = new Random();

        private string GenerateText(string seed, int length, double temperature)
        {
            // 1. Zera o StringBuilder para uma nova geração
            generatedTextBuilder.Clear();

            // 2. Prepara a janela inicial de tokens a partir do 'seed'
            List<string> currentTokens = TokenizeTextForWindow(seed);
            while (currentTokens.Count < contextWindowSize)
                currentTokens.Insert(0, padToken);
            if (currentTokens.Count > contextWindowSize)
                currentTokens = currentTokens.Skip(currentTokens.Count - contextWindowSize).ToList();

            // 3. Loop de Geração
            for (int i = 0; i < length; i++)
            {
                // Verifica se o modelo não é nulo antes de usar
                if (model == null) throw new InvalidOperationException("O modelo não está carregado.");

                Tensor input = ConvertWindowToInputTensor(currentTokens);
                Tensor logitsTensor = model.ForwardLogits(input);
                double[] logits = logitsTensor.GetData();

                double[] probs = new double[logits.Length];
                double sumExp = 0;
                for (int j = 0; j < logits.Length; j++)
                {
                    double scaledLogit = logits[j] / temperature;
                    probs[j] = Math.Exp(scaledLogit);
                    sumExp += probs[j];
                }

                if (sumExp == 0 || double.IsNaN(sumExp) || double.IsInfinity(sumExp)) sumExp = 1e-9;
                    
                for (int j = 0; j < probs.Length; j++)
                    probs[j] /= sumExp;

                double r = rand.NextDouble();
                double cumulative = 0;
                int nextTokenIdx = 0;
                    
                for (int j = 0; j < probs.Length; j++)
                {
                    cumulative += probs[j];
                    if (r <= cumulative)
                    {
                        nextTokenIdx = j;
                        break;
                    }
                }

                string nextToken = indexToToken[nextTokenIdx];
                if (nextToken == padToken)
                {
                    // Se gerarmos um pad token, tentamos gerar novamente para não encurtar a resposta
                    i--; 
                    continue;
                }
                    
                generatedTextBuilder.Append(nextToken).Append(" ");
                    
                currentTokens.RemoveAt(0);
                currentTokens.Add(nextToken);
            }

            return generatedTextBuilder.ToString().Trim();
        }

        private void ExpandVocabularyAndAdaptModel(string newTextContent)
        {
            if (model == null)
            {
                Console.WriteLine("Aviso: Modelo não inicializado. Não é possível expandir vocabulário e adaptar.");
                return;
            }

            var newTokensFound = new HashSet<string>();
            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";

            var matches = Regex.Matches(newTextContent.ToLower(), pattern);
            foreach (Match match in matches)
            {
                string token = match.Value;
                if (string.IsNullOrEmpty(token) || char.IsControl(token[0]) && token[0] != ' ' || token == "\uFFFD" ||
                    (int)token[0] > 0x10FFFF)
                {
                    continue;
                }

                if (!tokenToIndex.ContainsKey(token))
                {
                    newTokensFound.Add(token);
                }
            }

            if (newTokensFound.Count == 0)
            {
                return;
            }

            Console.WriteLine($"Expandindo vocabulário com {newTokensFound.Count} novos tokens...");
            foreach (var newToken in newTokensFound.OrderBy(t => t))
            {
                if (!tokenToIndex.ContainsKey(newToken))
                {
                    tokenToIndex[newToken] = indexToToken.Count;
                    indexToToken.Add(newToken);
                }
            }

            int oldVocabSize = model.OutputSize;
            int newVocabSize = tokenToIndex.Count;

            if (newVocabSize > oldVocabSize)
            {
                Console.WriteLine(
                    $"Vocabulário expandido de {oldVocabSize} para {newVocabSize} tokens. Adaptando modelo...");

                var newModel = new NeuralNetwork(newVocabSize * contextWindowSize, HiddenSize, newVocabSize,
                    contextWindowSize);

                for (int h = 0; h < HiddenSize; h++)
                {
                    for (int oldVocabIdx = 0; oldVocabIdx < oldVocabSize; oldVocabIdx++)
                    {
                        for (int k = 0; k < contextWindowSize; k++)
                        {
                            int oldFlatIndex = (k * oldVocabSize + oldVocabIdx) * HiddenSize + h;
                            int newFlatIndex = (k * newVocabSize + oldVocabIdx) * HiddenSize + h;
                            if (oldFlatIndex < model.W_i_Tensor.GetData().Length &&
                                newFlatIndex < newModel.W_i_Tensor.GetData().Length)
                            {
                                double[] oldWiData = model.W_i_Tensor.GetData();
                                double[] newWiData = newModel.W_i_Tensor.GetData();
                                newWiData[newFlatIndex] = oldWiData[oldFlatIndex];
                                newModel.W_i_Tensor.SetData(newWiData);
                            }
                        }
                    }
                }

                newModel.U_i_Tensor.SetData(model.U_i_Tensor.GetData());
                newModel.U_f_Tensor.SetData(model.U_f_Tensor.GetData());
                newModel.U_c_Tensor.SetData(model.U_c_Tensor.GetData());
                newModel.U_o_Tensor.SetData(model.U_o_Tensor.GetData());

                newModel.b_i_Tensor.SetData(model.b_i_Tensor.GetData());
                newModel.b_f_Tensor.SetData(model.b_f_Tensor.GetData());
                newModel.b_c_Tensor.SetData(model.b_c_Tensor.GetData());
                newModel.b_o_Tensor.SetData(model.b_o_Tensor.GetData());

                for (int h = 0; h < HiddenSize; h++)
                {
                    for (int oldVocabIdx = 0; oldVocabIdx < oldVocabSize; oldVocabIdx++)
                    {
                        int oldFlatIndex = h * oldVocabSize + oldVocabIdx;
                        int newFlatIndex = h * newVocabSize + oldVocabIdx;
                        if (oldFlatIndex < model.W_out_Tensor.GetData().Length &&
                            newFlatIndex < newModel.W_out_Tensor.GetData().Length)
                        {
                            double[] oldWoutData = model.W_out_Tensor.GetData();
                            double[] newWoutData = newModel.W_out_Tensor.GetData();
                            newWoutData[newFlatIndex] = oldWoutData[oldFlatIndex];
                            newModel.W_out_Tensor.SetData(newWoutData);
                        }
                    }
                }

                for (int oldVocabIdx = 0; oldVocabIdx < oldVocabSize; oldVocabIdx++)
                {
                    if (oldVocabIdx < model.b_out_Tensor.GetData().Length &&
                        oldVocabIdx < newModel.b_out_Tensor.GetData().Length)
                    {
                        double[] oldBoutData = model.b_out_Tensor.GetData();
                        double[] newBoutData = newModel.b_out_Tensor.GetData();
                        newBoutData[oldVocabIdx] = oldBoutData[oldVocabIdx];
                        newModel.b_out_Tensor.SetData(newBoutData);
                    }
                }

                model?.Dispose();
                model = newModel;
                Console.WriteLine("Modelo adaptado com sucesso ao novo vocabulário.");

                SaveVocabulary();
            }
            else if (newVocabSize == oldVocabSize)
            {
                Console.WriteLine("Vocabulário não expandiu, nenhuma adaptação de modelo necessária.");
            }
        }

        [HttpPost("summarize")]
        public async Task<IActionResult> Summarize([FromBody] SummaryRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.TextToSummarize))
                {
                    return BadRequest(new { Error = "TextToSummarize não pode estar vazio." });
                }

                string generatedSummary;

                Console.WriteLine("Solicitando resumo inteligente ao serviço de aquisição de conhecimento...");
                var summaryContentParts = await _knowledgeAcquisitionService.GetSummarizationFromExternalService(
                    $"Summarize the following text: {request.TextToSummarize}", request.SummaryLengthWords * 2 ?? 500);

                generatedSummary = string.Join(" ", summaryContentParts);

                if (string.IsNullOrEmpty(generatedSummary))
                {
                    Console.WriteLine("Serviço externo falhou em gerar o resumo. Usando o resumo local.");
                    generatedSummary =
                        _textProcessorService.Summarize(request.TextToSummarize, request.SummaryLengthWords ?? 100);
                }

                ExpandVocabularyAndAdaptModel(request.TextToSummarize);

                string contextTopic = _textProcessorService.ExtractMainTopic(request.TextToSummarize);
                ContextInfo summaryContext = new ContextInfo
                {
                    ContextId = _textProcessorService.GenerateContextHash(contextTopic),
                    Topic = contextTopic,
                    Summary = generatedSummary,
                    Urls = request.SourceUrls ?? new List<string>(),
                    ExternalLastUpdatedTicks = DateTime.UtcNow.Ticks
                };

                byte[] serializedData =
                    Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(summaryContext));
                if (serializedData.Length > TreeNode.MaxDataSize)
                {
                    Array.Resize(ref serializedData, TreeNode.MaxDataSize);
                }

                long existingOffset = _contextManager.FindContextOffsetByTopic(contextTopic);
                if (existingOffset != -1)
                {
                    Console.WriteLine($"Contexto de resumo existente para '{contextTopic}'. Atualizando...");
                    _memoryStorage.UpdateData(existingOffset, Encoding.UTF8.GetString(serializedData));
                }
                else
                {
                    Console.WriteLine($"Novo contexto de resumo para '{contextTopic}'. Inserindo...");
                    _memoryStorage.Insert(Encoding.UTF8.GetString(serializedData));
                }

                return Ok(new { Summary = generatedSummary, ContextStored = true });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { Error = $"Falha ao gerar resumo: {ex.Message}" });
            }
        }

        [HttpPost("evaluate")]
        public IActionResult Evaluate([FromBody] TestRequest request)
        {
            try
            {
                if (model == null || tokenToIndex.Count == 0)
                {
                    return BadRequest(new
                        { Error = "Modelo ou vocabulário não inicializados. Treine o modelo primeiro." });
                }

                if (request.ContextWindowSize != contextWindowSize)
                {
                    return BadRequest(new
                    {
                        Error =
                            $"ContextWindowSize da requisição ({request.ContextWindowSize}) deve ser igual ao ContextWindowSize do modelo carregado ({contextWindowSize})."
                    });
                }

                if (request.ContextWindowSize <= 0)
                {
                    return BadRequest(new { Error = "ContextWindowSize deve ser positivo." });
                }

                if (string.IsNullOrEmpty(request.TextData))
                {
                    return BadRequest(new { Error = "TextData não pode estar vazio." });
                }

                if (!IsValidText(request.TextData))
                {
                    return BadRequest(new
                        { Error = "O texto de entrada contém tokens não presentes no vocabulário de treinamento." });
                }

                string seed = request.TextData.ToLower();
                int length = 50;
                double temperature = 1.0;

                StringBuilder generatedText = new StringBuilder(seed);

                List<string> currentTokens = TokenizeTextForWindow(seed);
                while (currentTokens.Count < contextWindowSize)
                {
                    currentTokens.Insert(0, padToken);
                }

                if (currentTokens.Count > contextWindowSize)
                {
                    currentTokens = currentTokens.Skip(currentTokens.Count - contextWindowSize).ToList();
                }

                Random rand = new Random();
                var specialChars = new[] { ".", ",", "!", "?", ":", ";", "\"", "'", "-", "(", ")" };

                for (int i = 0; i < length; i++)
                {
                    Tensor input = ConvertWindowToInputTensor(currentTokens);
                    Tensor logitsTensor = model.ForwardLogits(input);
                    double[] logits = logitsTensor.GetData();

                    double[] probs = new double[logits.Length];
                    double sumExpTemp = 0;
                    for (int j = 0; j < logits.Length; j++)
                    {
                        probs[j] = Math.Exp(logits[j] / temperature);
                        sumExpTemp += probs[j];
                    }

                    for (int j = 0; j < probs.Length; j++)
                    {
                        probs[j] /= sumExpTemp;
                    }

                    double r = rand.NextDouble() * probs.Sum();
                    double cumulative = 0;
                    int nextTokenIdx = 0;
                    for (int j = 0; j < probs.Length; j++)
                    {
                        cumulative += probs[j];
                        if (r <= cumulative)
                        {
                            nextTokenIdx = j;
                            break;
                        }
                    }

                    string nextToken = indexToToken[nextTokenIdx];

                    if (nextToken == padToken) continue;

                    bool isSpecialChar = specialChars.Contains(nextToken);
                    bool lastCharIsSpecialChar =
                        generatedText.Length > 0 && specialChars.Contains(generatedText[^1].ToString());

                    if (!isSpecialChar)
                    {
                        if (generatedText.Length > 0 && generatedText[^1] != ' ' && !lastCharIsSpecialChar)
                        {
                            generatedText.Append(" ");
                        }
                    }
                    else
                    {
                        if (generatedText.Length > 0 && generatedText[^1] == ' ')
                        {
                            generatedText.Remove(generatedText.Length - 1, 1);
                        }
                    }

                    generatedText.Append(nextToken);
                    currentTokens.RemoveAt(0);
                    currentTokens.Add(nextToken);
                }

                string finalGeneratedText = generatedText.ToString().Trim();
                if (finalGeneratedText.Length > 0 && char.IsLetter(finalGeneratedText[0]))
                {
                    finalGeneratedText = char.ToUpper(finalGeneratedText[0]) + finalGeneratedText.Substring(1);
                }

                return Ok(new { EvaluatedText = finalGeneratedText });
            }
            catch (Exception ex)
            {
                return BadRequest(new { Error = $"Falha na avaliação: {ex.Message}" });
            }
        }

        private void BuildVocabulary(string text)
        {
            tokenToIndex.Clear();
            indexToToken.Clear();
            tokenToIndex[padToken] = indexToToken.Count;
            indexToToken.Add(padToken);

            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";

            var matches = Regex.Matches(text.ToLower(), pattern);
            var tokens = matches.Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).Distinct().OrderBy(t => t)
                .ToArray();

            foreach (string token in tokens)
            {
                if (char.IsControl(token[0]) && token[0] != ' ' || token == "\uFFFD" ||
                    (int)token[0] > 0x10FFFF) continue;

                if (!tokenToIndex.ContainsKey(token))
                {
                    tokenToIndex[token] = indexToToken.Count;
                    indexToToken.Add(token);
                }
            }

            if (tokenToIndex.Count > 0)
            {
                SaveVocabulary();
            }
        }

        private void SaveVocabulary()
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

                Console.WriteLine($"Vocabulário salvo em: {vocabPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao salvar vocabulário: {ex.Message}");
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

        public List<string> TokenizeTextForWindow(string text)
        {
            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            var matches = Regex.Matches(text.ToLower(), pattern);
            var tokens = matches.Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToList();

            for (int i = 0; i < tokens.Count; i++)
            {
                if (!tokenToIndex.ContainsKey(tokens[i]))
                {
                    tokens[i] = padToken;
                }
            }

            return tokens;
        }

        public Tensor ConvertWindowToInputTensor(List<string> windowTokens)
        {
            double[] inputData = new double[tokenToIndex.Count * contextWindowSize];
            for (int k = 0; k < contextWindowSize; k++)
            {
                string token = windowTokens[k];
                int tokenVocabIndex = tokenToIndex.ContainsKey(token) ? tokenToIndex[token] : tokenToIndex[padToken];
                int offset = k * tokenToIndex.Count;
                inputData[offset + tokenVocabIndex] = 1.0;
            }

            return new Tensor(inputData, new int[] { tokenToIndex.Count * contextWindowSize });
        }
    }
}
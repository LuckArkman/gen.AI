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
using Services; // CORRIGIDO: Namespace correto para seus serviços customizados

namespace GenerativeAIAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class GenerativeAIController : ControllerBase
    {
        private readonly string modelDir; // Agora readonly
        private readonly string modelPath;
        private readonly string vocabPath;
        private NeuralNetwork? model;
        private Dictionary<string, int> tokenToIndex;
        private List<string> indexToToken;
        private const int HiddenSize = 256;
        private readonly string padToken = "[PAD]";
        private readonly int contextWindowSize;
        private BinaryTreeFileStorage _memoryStorage;
        private readonly TextProcessorService _textProcessorService;
        private readonly string _memoryFilePath;
        private readonly ContextManager _contextManager;
        // private readonly ChatGPTService _chatGPTService; // REMOVIDO: Gerenciado por KnowledgeAcquisitionService
        private readonly KnowledgeAcquisitionService _knowledgeAcquisitionService;
        // private readonly GenerateRequest _generateRequest; // REMOVIDO: Usar o parâmetro 'request' do método Generate

        private const double KnowledgeInternalizationLearningRate = 0.00001;

        public GenerativeAIController(IConfiguration configuration,
            ContextManager contextManager,
            TextProcessorService textProcessorService,
            KnowledgeAcquisitionService knowledgeAcquisitionService)
        {
            modelDir = configuration["ModelSettings:ModelDirectory"] ?? "/home/mplopes/Documentos/generative/generative/"; // CORRIGIDO: Inicializa modelDir primeiro

            _contextManager = contextManager;
            _textProcessorService = textProcessorService;
            _knowledgeAcquisitionService = knowledgeAcquisitionService;

            _memoryFilePath = configuration["ModelSettings:MemoryFilePath"] ?? Path.Combine(modelDir, "AIModelMem.dat");
            _memoryStorage = new BinaryTreeFileStorage(_memoryFilePath);
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

        private double InternalizeKnowledgeIntoModel(string knowledgeText)
        {
            if (model == null || tokenToIndex.Count == 0)
            {
                Console.WriteLine("Modelo ou vocabulário não inicializados para internalizar conhecimento.");
                return 0;
            }

            var (inputs, targets) = PrepareDataset(knowledgeText, contextWindowSize);
            if (inputs.Length == 0)
            {
                Console.WriteLine("Dados de conhecimento insuficientes para internalização.");
                return 0;
            }

            Console.WriteLine($"Internalizando {inputs.Length} sequências de conhecimento no modelo...");
            double loss = model.TrainEpoch(inputs, targets, KnowledgeInternalizationLearningRate);
            Console.WriteLine($"Perda na internalização de conhecimento: {loss:F4}");
            model.SaveModel(modelPath);
            return loss;
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
                int requestContextWindowSize = request.ContextWindowSize;

                if (string.IsNullOrEmpty(request.TextData))
                {
                    return BadRequest(new { Error = "TextData não pode estar vazio." });
                }

                if (requestContextWindowSize <= 0)
                {
                    return BadRequest(new { Error = "ContextWindowSize deve ser positivo." });
                }

                if (request.LearningRate.HasValue && request.LearningRate <= 0)
                {
                    return BadRequest(new { Error = "LearningRate deve ser positivo." });
                }

                if (request.Epochs.HasValue && request.Epochs <= 0)
                {
                    return BadRequest(new { Error = "Epochs deve ser positivo." });
                }

                if (System.IO.File.Exists(vocabPath))
                {
                    LoadVocabulary();
                }

                if (tokenToIndex.Count == 0)
                {
                    BuildVocabulary(request.TextData);
                }

                int vocabSize = tokenToIndex.Count;
                if (vocabSize == 0)
                {
                    return BadRequest(new { Error = "Nenhum token válido encontrado no texto de treinamento." });
                }

                if (model == null || model.InputSize != vocabSize * requestContextWindowSize ||
                    model.OutputSize != vocabSize)
                {
                    Console.WriteLine(
                        $"Inicializando novo modelo com VocabSize: {vocabSize}, ContextWindowSize: {requestContextWindowSize}");
                    model = new NeuralNetwork(vocabSize * requestContextWindowSize, HiddenSize, vocabSize, requestContextWindowSize);
                }

                var (inputs, targets) = PrepareDataset(request.TextData, requestContextWindowSize);

                if (inputs.Length == 0)
                {
                    return BadRequest(new
                        { Error = "Dados de treinamento insuficientes para a ContextWindowSize especificada." });
                }

                double learningRate = request.LearningRate ?? 0.01;
                int epochs = request.Epochs ?? 10;
                double totalLoss = 0;
                var losses = new List<double>();

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double epochLoss = model.TrainEpoch(inputs, targets, learningRate);
                    losses.Add(epochLoss);
                    totalLoss += epochLoss;
                    Console.WriteLine($"Época {epoch + 1}/{epochs}, Perda: {epochLoss:F4}");
                }

                model.SaveModel(modelPath);
                SaveVocabulary();

                return Ok(new
                {
                    Message = "Treinamento concluído", AverageLoss = totalLoss / epochs, VocabularySize = vocabSize,
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
                        { Error = "Os dados de teste contêm tokens não presentes no vocabulário de treinamento." });
                }

                var (inputs, targets) = PrepareDataset(request.TextData, request.ContextWindowSize);

                if (inputs.Length == 0)
                {
                    return BadRequest(new
                        { Error = "Dados de teste insuficientes para a ContextWindowSize especificada." });
                }

                double totalLoss = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Tensor output = model.Forward(inputs[i]);
                    for (int o = 0; o < tokenToIndex.Count; o++)
                    {
                        if (targets[i].Infer(new int[] { o }) == 1.0)
                        {
                            double outputValue = output.Infer(new int[] { o });
                            totalLoss += -Math.Log(outputValue + 1e-9);
                            break;
                        }
                    }
                }

                double averageLoss = inputs.Length > 0 ? totalLoss / inputs.Length : 0;
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
            Console.WriteLine($"UserInput : {request.SeedText} >> SequenceLength :" +
                              $" {request.SequenceLength} >> Length : {request.Length} >> Temperature : {request.Temperature} >>" +
                              $"ContextWindowSize : {request.ContextWindowSize}");
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

                if (request.Length.HasValue && request.Length <= 0)
                {
                    return BadRequest(new { Error = "Length deve ser positivo." });
                }

                if (request.Temperature <= 0)
                {
                    return BadRequest(new { Error = "Temperature deve ser positivo." });
                }

                if (string.IsNullOrEmpty(request.SeedText))
                {
                    return BadRequest(new { Error = "SeedText não pode estar vazio." });
                }

                // 1. Reflexão e Enriquecimento:
                string enrichedContext = string.Empty;
                string topic = _textProcessorService.ExtractMainTopic(request.SeedText);
                ContextInfo? storedContext = _contextManager.GetContextByTopic(topic);

                bool needsExternalAcquisition = false;
                if (storedContext == null)
                {
                    Console.WriteLine(
                        $"Reflexão: Nenhum contexto para '{topic}' na memória. Iniciando aquisição de nova informação externa.");
                    needsExternalAcquisition = true;
                }
                else
                {
                    DateTime lastUpdated = new DateTime(storedContext.ExternalLastUpdatedTicks);
                    if (DateTime.UtcNow - lastUpdated > _contextManager.MaxContextAgeForRefresh)
                    {
                        Console.WriteLine(
                            $"Reflexão: Contexto para '{topic}' desatualizado. Buscando informação externa para atualização.");
                        needsExternalAcquisition = true;
                    }
                    else
                    {
                        enrichedContext = storedContext.Summary;
                        Console.WriteLine(
                            $"Contexto recuperado da memória virtual (atualizado): {storedContext.Summary}");
                    }
                }

                if (needsExternalAcquisition)
                {
                    var (externalContent, sourceName) = await _knowledgeAcquisitionService.GetInformation(topic);

                    if (externalContent.Any() && !string.IsNullOrWhiteSpace(string.Join("", externalContent)))
                    {
                        string combinedExternalContent = string.Join(" ", externalContent);

                        ExpandVocabularyAndAdaptModel(combinedExternalContent);

                        string newSummary = _textProcessorService.Summarize(combinedExternalContent);
                        InternalizeKnowledgeIntoModel(newSummary);

                        ContextInfo contextToStore = storedContext ?? new ContextInfo
                        {
                            ContextId = _textProcessorService.GenerateContextHash(topic),
                            Topic = topic,
                            Urls = new List<string> { sourceName }
                        };

                        if (storedContext == null || !string.Equals(newSummary.Trim(), storedContext.Summary.Trim(),
                                StringComparison.OrdinalIgnoreCase))
                        {
                            Console.WriteLine(
                                $"Informação relevante obtida de {sourceName}. Atualizando/Inserindo memória virtual.");
                            contextToStore.Summary = newSummary;
                            contextToStore.ExternalLastUpdatedTicks = DateTime.UtcNow.Ticks;

                            byte[] serializedData = Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(contextToStore));
                            if (serializedData.Length > TreeNode.MaxDataSize)
                            {
                                Console.WriteLine(
                                    $"Aviso: Contexto externo muito grande ({serializedData.Length} bytes). Truncando para {TreeNode.MaxDataSize} bytes antes de armazenar.");
                                Array.Resize(ref serializedData, TreeNode.MaxDataSize);
                            }

                            long offsetToUpdateOrInsert = _contextManager.FindContextOffsetByTopic(topic);
                            if (offsetToUpdateOrInsert != -1)
                            {
                                _memoryStorage.UpdateData(offsetToUpdateOrInsert,
                                    Encoding.UTF8.GetString(serializedData));
                            }
                            else
                            {
                                _memoryStorage.Insert(Encoding.UTF8.GetString(serializedData));
                            }

                            _memoryStorage.CleanUnusedNodes(TimeSpan.FromDays(30));
                        }
                        else
                        {
                            Console.WriteLine(
                                $"Informação de {sourceName} não é significativamente diferente ou é redundante. Não atualizando o contexto existente.");
                        }

                        enrichedContext = newSummary;
                    }
                    else
                    {
                        Console.WriteLine("Nenhuma fonte externa retornou conteúdo relevante.");
                        if (storedContext != null) enrichedContext = storedContext.Summary;
                    }
                }
                // --- FIM DO FLUXO DE REFLEXÃO E ENRIQUECIMENTO ---

                StringBuilder generatedTextBuilder = new StringBuilder();

                string effectiveSeed = string.IsNullOrEmpty(enrichedContext)
                    ? request.SeedText
                    : $"{request.SeedText}. Contexto relevante: {enrichedContext}";
                
                generatedTextBuilder.Append(effectiveSeed);

                List<string> currentTokens = TokenizeTextForWindow(effectiveSeed);
                while (currentTokens.Count < request.ContextWindowSize)
                    currentTokens.Insert(0, padToken);
                if (currentTokens.Count > request.ContextWindowSize)
                    currentTokens = currentTokens.Skip(currentTokens.Count - request.ContextWindowSize).ToList();

                Random rand = new Random();
                var specialChars = new[] { ".", ",", "!", "?", ":", ";", "\"", "'", "-", "(", ")" };

                for (int i = 0; i < (request.Length ?? 50); i++)
                {
                    Tensor input = ConvertWindowToInputTensor(currentTokens);
                    Tensor logitsTensor = model.ForwardLogits(input);
                    double[] logits = logitsTensor.GetData();

                    double[] probs = new double[logits.Length];
                    double sumExpTemp = 0;
                    for (int j = 0; j < logits.Length; j++)
                    {
                        probs[j] = Math.Exp(logits[j] / request.Temperature);
                        sumExpTemp += probs[j];
                    }

                    for (int j = 0; j < probs.Length; j++)
                        probs[j] /= sumExpTemp;

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
                        generatedTextBuilder.Length > 0 && specialChars.Contains(generatedTextBuilder[^1].ToString());

                    if (!isSpecialChar && generatedTextBuilder.Length > 0 && generatedTextBuilder[^1] != ' ' && !lastCharIsSpecialChar)
                    {
                        generatedTextBuilder.Append(" ");
                    }
                    else if (isSpecialChar && generatedTextBuilder.Length > 0 && generatedTextBuilder[^1] == ' ')
                    {
                        generatedTextBuilder.Length--;
                    }

                    generatedTextBuilder.Append(nextToken);

                    if (isSpecialChar && nextToken != "\"" && nextToken != "'" && nextToken != "(" && nextToken != ")")
                    {
                        generatedTextBuilder.Append(" ");
                    }

                    currentTokens.RemoveAt(0);
                    currentTokens.Add(nextToken);
                }

                string finalGeneratedText = generatedTextBuilder.ToString().Trim();
                if (finalGeneratedText.Length > 0 && char.IsLetter(finalGeneratedText[0]))
                    finalGeneratedText = char.ToUpper(finalGeneratedText[0]) + finalGeneratedText.Substring(1);

                await _contextManager.StoreConversationContext(request.SeedText, finalGeneratedText);
                Console.WriteLine($"AI response : {finalGeneratedText}");

                return Ok(finalGeneratedText);
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { Error = $"Falha na geração: {ex.Message}" });
            }
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
                    if (oldVocabIdx < model.b_out_Tensor.GetData().Length && oldVocabIdx < newModel.b_out_Tensor.GetData().Length)
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
                InternalizeKnowledgeIntoModel(generatedSummary);

                string contextTopic = _textProcessorService.ExtractMainTopic(request.TextToSummarize);
                ContextInfo summaryContext = new ContextInfo
                {
                    ContextId = _textProcessorService.GenerateContextHash(contextTopic),
                    Topic = contextTopic,
                    Summary = generatedSummary,
                    Urls = request.SourceUrls ?? new List<string>(),
                    ExternalLastUpdatedTicks = DateTime.UtcNow.Ticks
                };

                byte[] serializedData = Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(summaryContext));
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

                inputs.Add(new Tensor(inputData, new int[] { tokenToIndex.Count * currentContextWindowSize }));

                double[] targetData = new double[tokenToIndex.Count];
                targetData[tokenToIndex[nextToken]] = 1.0;
                targets.Add(new Tensor(targetData, new int[] { tokenToIndex.Count }));
            }

            return (inputs.ToArray(), targets.ToArray());
        }

        private List<string> TokenizeTextForWindow(string text)
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

        private Tensor ConvertWindowToInputTensor(List<string> windowTokens)
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
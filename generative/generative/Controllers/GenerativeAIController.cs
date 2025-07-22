using Microsoft.AspNetCore.Mvc;
using System.Text;
using System.Text.RegularExpressions;
using Core;
using Models;
using Microsoft.Extensions.Configuration;
using System.IO;
using System.Collections.Generic;
using System;
using System.Linq; // Adicionado para .Last() e .Skip()

namespace GenerativeAIAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class GenerativeAIController : ControllerBase
    {
        private readonly string modelDir;
        private readonly string modelPath;
        private readonly string vocabPath;
        private NeuralNetwork? model;
        private Dictionary<string, int> tokenToIndex;
        private List<string> indexToToken;
        private const int HiddenSize = 256;
        private readonly string padToken = "[PAD]";
        private readonly int contextWindowSize; // Novo: Armazenará o tamanho da janela de contexto

        public GenerativeAIController(IConfiguration configuration)
        {
            modelDir = "/home/mplopes/Documentos/generative/generative/";
            
            if (!Directory.Exists(modelDir))
            {
                Console.WriteLine($"Aviso: O diretório do modelo '{modelDir}' não existe na inicialização da API.");
            }

            // O número da época final será lido da configuração do aplicativo.
            contextWindowSize = configuration.GetValue<int>("ModelSettings:ContextWindowSize", 10); // Novo: Padrão para 10
            
            modelPath = Path.Combine(modelDir, $"model.json"); // Carrega o modelo da época final
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
                        // Verifica se o modelo carregado é compatível com o VOCABULÁRIO e o TAMANHO DA JANELA
                        if (model != null && 
                            (model.InputSize != tokenToIndex.Count * contextWindowSize || model.OutputSize != tokenToIndex.Count))
                        {
                            Console.WriteLine($"Modelo carregado, mas suas dimensões ({model.InputSize}, {model.OutputSize}) não correspondem ao tamanho do vocabulário ({tokenToIndex.Count}) e ContextWindowSize ({contextWindowSize}). O modelo pode ser incompatível.");
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
                Console.WriteLine("Modelo ou vocabulário não encontrados na inicialização do controlador. Treine o modelo primeiro.");
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
                // Este endpoint é mantido para compatibilidade, mas o treinamento completo deve usar o programa Trainer.
                // Ajustamos ContextWindowSize aqui para que ele use o que foi passado na requisição, se houver.
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

                // Inicializa o modelo com o novo input size (vocabSize * contextWindowSize)
                if (model == null || model.InputSize != vocabSize * requestContextWindowSize || model.OutputSize != vocabSize)
                {
                    Console.WriteLine($"Inicializando novo modelo com VocabSize: {vocabSize}, ContextWindowSize: {requestContextWindowSize}");
                    model = new NeuralNetwork(vocabSize * requestContextWindowSize, HiddenSize, vocabSize);
                }

                var (inputs, targets) = PrepareDataset(request.TextData, requestContextWindowSize);
                
                if (inputs.Length == 0)
                {
                     return BadRequest(new { Error = "Dados de treinamento insuficientes para a ContextWindowSize especificada." });
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

                return Ok(new { Message = "Treinamento concluído", AverageLoss = totalLoss / epochs, VocabularySize = vocabSize, EpochLosses = losses });
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
                // Usa o ContextWindowSize da requisição, mas o modelo foi treinado com o da API.
                // Para consistência, é melhor que o ContextWindowSize da requisição seja o mesmo do modelo carregado.
                // Por isso, fazemos uma validação.
                if (request.ContextWindowSize != contextWindowSize)
                {
                     return BadRequest(new { Error = $"ContextWindowSize da requisição ({request.ContextWindowSize}) deve ser igual ao ContextWindowSize do modelo carregado ({contextWindowSize})." });
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
                    return BadRequest(new { Error = "Os dados de teste contêm tokens não presentes no vocabulário de treinamento." });
                }

                var (inputs, targets) = PrepareDataset(request.TextData, request.ContextWindowSize);
                
                if (inputs.Length == 0)
                {
                    return BadRequest(new { Error = "Dados de teste insuficientes para a ContextWindowSize especificada." });
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
        public IActionResult Generate([FromBody] GenerateRequest request)
        {
            try
            {
                Console.WriteLine($"Input do usuario {request.SeedText}: SequenceLength {request.SequenceLength}, SeedText {request.SeedText}, ContextWindowSize {request.ContextWindowSize}, Length {request.Length}, Temperature {request.Temperature}");
                if (model == null || tokenToIndex.Count == 0)
                {
                    return BadRequest(new { Error = "Modelo ou vocabulário não inicializados. Treine o modelo primeiro." });
                }
                // Valida ContextWindowSize
                if (request.ContextWindowSize != contextWindowSize)
                {
                     return BadRequest(new { Error = $"ContextWindowSize da requisição ({request.ContextWindowSize}) deve ser igual ao ContextWindowSize do modelo carregado ({contextWindowSize})." });
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

                if (!IsValidText(request.SeedText))
                {
                    return BadRequest(new { Error = "O texto semente contém tokens não presentes no vocabulário de treinamento." });
                }

                string seed = string.IsNullOrEmpty(request.SeedText) ? padToken : request.SeedText.ToLower();
                int length = request.Length ?? 50;
                double temperature = request.Temperature;

                StringBuilder generatedText = new StringBuilder(seed);

                // Converte a semente em uma lista de tokens, preenchendo com [PAD] se necessário
                // até atingir o tamanho da janela de contexto.
                List<string> currentTokens = TokenizeTextForWindow(seed);
                
                // Reduz a lista para ter apenas o ContextWindowSize de tokens (os últimos da semente)
                // e garante que sempre tenha ContextWindowSize tokens, preenchendo com PAD no início se faltar.
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
                    
                    if (nextToken == padToken)
                    {
                        // Se o modelo prevê PAD, podemos parar a geração ou ignorá-lo
                        // Por enquanto, vamos ignorar e continuar, mas é um sinal de que pode ser o fim da "ideia"
                        continue; 
                    }

                    bool isSpecialChar = specialChars.Contains(nextToken);
                    bool lastCharIsSpecialChar = generatedText.Length > 0 && specialChars.Contains(generatedText[^1].ToString());

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
                    
                    // Atualiza a janela de contexto para a próxima previsão
                    currentTokens.RemoveAt(0); // Remove o token mais antigo
                    currentTokens.Add(nextToken); // Adiciona o token recém-gerado
                }

                string finalGeneratedText = generatedText.ToString().Trim();
                if (finalGeneratedText.Length > 0 && char.IsLetter(finalGeneratedText[0]))
                {
                    finalGeneratedText = char.ToUpper(finalGeneratedText[0]) + finalGeneratedText.Substring(1);
                }
                Console.WriteLine($"Output do Modelo {finalGeneratedText}");
                return Ok(new { GeneratedText = finalGeneratedText });
            }
            catch (Exception ex)
            {
                return BadRequest(new { Error = $"Falha na geração: {ex.Message}" });
            }
        }

        [HttpPost("evaluate")]
        public IActionResult Evaluate([FromBody] TestRequest request)
        {
            try
            {
                if (model == null || tokenToIndex.Count == 0)
                {
                    return BadRequest(new { Error = "Modelo ou vocabulário não inicializados. Treine o modelo primeiro." });
                }
                // Valida ContextWindowSize
                if (request.ContextWindowSize != contextWindowSize)
                {
                     return BadRequest(new { Error = $"ContextWindowSize da requisição ({request.ContextWindowSize}) deve ser igual ao ContextWindowSize do modelo carregado ({contextWindowSize})." });
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
                    return BadRequest(new { Error = "O texto de entrada contém tokens não presentes no vocabulário de treinamento." });
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
                    bool lastCharIsSpecialChar = generatedText.Length > 0 && specialChars.Contains(generatedText[^1].ToString());

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
            var tokens = matches.Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).Distinct().OrderBy(t => t).ToArray();

            foreach (string token in tokens)
            {
                if (char.IsControl(token[0]) && token[0] != ' ' || token == "\uFFFD" || (int)token[0] > 0x10FFFF) continue;

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
                            Console.WriteLine($"Linha inválida ignorada no vocabulário na linha {lineNumber}: '{line}'");
                            continue;
                        }

                        string token = line;
                        if (token == padToken && tokenToIndex.ContainsKey(padToken)) continue; 

                        if (char.IsControl(token[0]) && token[0] != ' ' || token == "\uFFFD" || (int)token[0] > 0x10FFFF)
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

        // Modificado para aceitar contextWindowSize
        private (Tensor[] inputs, Tensor[] targets) PrepareDataset(string text, int currentContextWindowSize)
        {
            var inputs = new List<Tensor>();
            var targets = new List<Tensor>();

            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            
            var matches = Regex.Matches(text.ToLower(), pattern);
            var tokens = matches.Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToArray();

            // Adiciona padding no INÍCIO da sequência para formar as primeiras janelas de contexto
            var paddedTokens = new List<string>();
            for (int k = 0; k < currentContextWindowSize; k++)
            {
                paddedTokens.Add(padToken);
            }
            paddedTokens.AddRange(tokens);

            // O loop vai até o ponto em que ainda há tokens suficientes para uma janela completa + o token alvo.
            for (int i = 0; i < paddedTokens.Count - currentContextWindowSize; i++)
            {
                string[] currentWindowTokens = paddedTokens.Skip(i).Take(currentContextWindowSize).ToArray();
                string nextToken = paddedTokens[i + currentContextWindowSize];

                if (!tokenToIndex.ContainsKey(nextToken) || !currentWindowTokens.All(t => tokenToIndex.ContainsKey(t)))
                {
                    Console.WriteLine($"Sequência ignorada no dataset (índice {i}): tokens ausentes no vocabulário. Próximo Token: '{nextToken}', Janela: '{string.Join(" ", currentWindowTokens)}'");
                    continue;
                }

                // Cria o tensor de entrada como uma concatenação de one-hot vectors
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

        // Novo método para tokenizar texto e preencher janela para geração
        private List<string> TokenizeTextForWindow(string text)
        {
            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            var matches = Regex.Matches(text.ToLower(), pattern);
            var tokens = matches.Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToList();

            // Substitui tokens fora do vocabulário por [PAD]
            for (int i = 0; i < tokens.Count; i++)
            {
                if (!tokenToIndex.ContainsKey(tokens[i]))
                {
                    tokens[i] = padToken;
                }
            }
            return tokens;
        }

        // Novo método para converter a lista de tokens da janela em um tensor de entrada
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
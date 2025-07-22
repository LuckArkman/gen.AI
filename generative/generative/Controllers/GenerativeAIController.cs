using Microsoft.AspNetCore.Mvc;
using System.Text;
using System.Text.Json;
using Core;
using Models;

namespace GenerativeAIAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class GenerativeAIController : ControllerBase
    {
        private static string modelDir = "/home/mplopes/RiderProjects/generative/generative/models/";
        private readonly string modelPath = Path.Combine(modelDir, "model_epoch_10.json");
        private readonly string vocabPath = Path.Combine(modelDir, "vocab_epoch_10.txt");
        private NeuralNetwork? model;
        private Dictionary<char, int> charToIndex;
        private List<char> indexToChar;
        private const int HiddenSize = 256;

        public GenerativeAIController()
        {
            charToIndex = new Dictionary<char, int>();
            indexToChar = new List<char>();
            if (System.IO.File.Exists(modelPath) && System.IO.File.Exists(vocabPath))
            {
                model = NeuralNetwork.LoadModel(modelPath);
                LoadVocabulary();
            }
        }

        private bool IsValidText(string? text)
        {
            return !string.IsNullOrEmpty(text);
        }

        // POST: api/GenerativeAI/train
        [HttpPost("train")]
        public IActionResult Train([FromBody] TrainRequest request)
        {
            try
            {
                if (string.IsNullOrEmpty(request.TextData))
                {
                    return BadRequest(new { Error = "TextData não pode estar vazio." });
                }
                if (request.SequenceLength <= 0)
                {
                    return BadRequest(new { Error = "SequenceLength deve ser positivo." });
                }
                if (request.LearningRate.HasValue && request.LearningRate <= 0)
                {
                    return BadRequest(new { Error = "LearningRate deve ser positivo." });
                }
                if (request.Epochs.HasValue && request.Epochs <= 0)
                {
                    return BadRequest(new { Error = "Epochs deve ser positivo." });
                }

                // Constrói vocabulário a partir do texto do CC-100
                BuildVocabulary(request.TextData);
                int vocabSize = charToIndex.Count;

                if (vocabSize == 0)
                {
                    return BadRequest(new { Error = "Nenhum caractere válido encontrado no texto de treinamento." });
                }

                // Inicializa o modelo com tamanho de entrada/saída baseado no vocabulário
                model = new NeuralNetwork(vocabSize, HiddenSize, vocabSize);

                // Prepara o conjunto de dados
                var (inputs, targets) = PrepareDataset(request.TextData, request.SequenceLength);
                
                // Treina o modelo
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

                // Salva o modelo e o vocabulário
                model.SaveModel(modelPath);
                SaveVocabulary();

                return Ok(new { Message = "Treinamento concluído", AverageLoss = totalLoss / epochs, VocabularySize = vocabSize, EpochLosses = losses });
            }
            catch (Exception ex)
            {
                return BadRequest(new { Error = $"Falha no treinamento: {ex.Message}" });
            }
        }

        // POST: api/GenerativeAI/test
        [HttpPost("test")]
        public IActionResult Test([FromBody] TestRequest request)
        {
            try
            {
                if (model == null || charToIndex.Count == 0)
                {
                    return BadRequest(new { Error = "Modelo ou vocabulário não inicializados. Treine o modelo primeiro." });
                }
                if (request.SequenceLength <= 0)
                {
                    return BadRequest(new { Error = "SequenceLength deve ser positivo." });
                }

                // Verifica se os dados de teste contêm caracteres fora do vocabulário
                if (!IsValidText(request.TextData))
                {
                    return BadRequest(new { Error = "Os dados de teste contêm caracteres não presentes no vocabulário de treinamento." });
                }

                // Prepara o conjunto de dados de teste
                var (inputs, targets) = PrepareDataset(request.TextData, request.SequenceLength);
                
                double totalLoss = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Tensor output = model.Forward(inputs[i]);
                    for (int o = 0; o < charToIndex.Count; o++)
                    {
                        if (targets[i].Infer(new int[] { o }) == 1.0)
                        {
                            totalLoss += -Math.Log(output.Infer(new int[] { o }) + 1e-9);
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

        // POST: api/GenerativeAI/generate
        [HttpPost("generate")]
        public IActionResult Generate([FromBody] GenerateRequest request)
        {
            try
            {
                if (model == null || charToIndex.Count == 0)
                {
                    return BadRequest(new { Error = "Modelo ou vocabulário não inicializados. Treine o modelo primeiro." });
                }
                if (request.SequenceLength <= 0)
                {
                    return BadRequest(new { Error = "SequenceLength deve ser positivo." });
                }
                if (request.Length.HasValue && request.Length <= 0)
                {
                    return BadRequest(new { Error = "Length deve ser positivo." });
                }

                // Verifica se o texto semente contém caracteres válidos
                if (!IsValidText(request.SeedText))
                {
                    return BadRequest(new { Error = "O texto semente contém caracteres não presentes no vocabulário de treinamento." });
                }

                string seed = string.IsNullOrEmpty(request.SeedText) ? indexToChar[0].ToString() : request.SeedText;
                int length = request.Length ?? 50;
                StringBuilder generatedText = new StringBuilder(seed);

                // Converte o texto semente para tensor de entrada (usa o último caractere)
                Tensor input = TextToTensor(seed);
                
                // Gera texto
                Random rand = new Random();
                for (int i = 0; i < length; i++)
                {
                    Tensor output = model.Forward(input);
                    double[] probs = output.GetData();
                    
                    // Amostra o próximo caractere usando probabilidades
                    double sum = probs.Sum();
                    double r = rand.NextDouble() * sum;
                    double cumulative = 0;
                    int nextCharIdx = 0;
                    for (int j = 0; j < probs.Length; j++)
                    {
                        cumulative += probs[j];
                        if (r <= cumulative)
                        {
                            nextCharIdx = j;
                            break;
                        }
                    }

                    char nextChar = indexToChar[nextCharIdx];
                    generatedText.Append(nextChar);

                    // Atualiza a entrada para a próxima iteração
                    input = CharToTensor(nextChar);
                }

                return Ok(new { GeneratedText = generatedText.ToString() });
            }
            catch (Exception ex)
            {
                return BadRequest(new { Error = $"Falha na geração: {ex.Message}" });
            }
        }

        private void BuildVocabulary(string text)
        {
            charToIndex = new Dictionary<char, int>();
            indexToChar = new List<char>();
            foreach (char c in text.Distinct().OrderBy(c => c))
            {
                // Ignora caracteres de controle, substituição (�) ou não imprimíveis
                if (char.IsControl(c) || c == '\uFFFD' || string.IsNullOrWhiteSpace(c.ToString())) continue;

                charToIndex[c] = indexToChar.Count;
                indexToChar.Add(c);
            }
        }

        private void SaveVocabulary()
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
                charToIndex = new Dictionary<char, int>();
                indexToChar = new List<char>();
                using (var reader = new StreamReader(vocabPath, Encoding.UTF8, true))
                {
                    while (!reader.EndOfStream)
                    {
                        string line = reader.ReadLine()?.Trim();
                        if (string.IsNullOrEmpty(line) || line.Length != 1) continue;
                        char c = line[0];
                        if (!charToIndex.ContainsKey(c) && !char.IsControl(c) && c != '\uFFFD' && !string.IsNullOrWhiteSpace(c.ToString()))
                        {
                            charToIndex[c] = indexToChar.Count;
                            indexToChar.Add(c);
                        }
                    }
                }
                if (indexToChar.Count == 0)
                {
                    throw new InvalidOperationException("Nenhum caractere válido encontrado no arquivo de vocabulário.");
                }
                Console.WriteLine($"Vocabulário carregado de: {vocabPath}, Tamanho: {charToIndex.Count} caracteres.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao carregar vocabulário: {ex.Message}");
                charToIndex = new Dictionary<char, int>();
                indexToChar = new List<char>();
            }
        }
        private (Tensor[] inputs, Tensor[] targets) PrepareDataset(string text, int sequenceLength)
        {
            // Divide o texto em parágrafos (formato CC-100)
            var paragraphs = text.Split(new[] { "\n\n" }, StringSplitOptions.RemoveEmptyEntries);
            var inputs = new List<Tensor>();
            var targets = new List<Tensor>();

            foreach (var paragraph in paragraphs)
            {
                var cleanParagraph = paragraph.Replace("\n", " ").Trim();
                if (cleanParagraph.Length < sequenceLength + 1) continue;

                for (int i = 0; i < cleanParagraph.Length - sequenceLength; i++)
                {
                    string sequence = cleanParagraph.Substring(i, sequenceLength);
                    char nextChar = cleanParagraph[i + sequenceLength];

                    if (!charToIndex.ContainsKey(nextChar) || !sequence.All(c => charToIndex.ContainsKey(c))) continue;

                    // Entrada: codifica one-hot o último caractere da sequência
                    double[] inputData = new double[charToIndex.Count];
                    inputData[charToIndex[sequence[sequenceLength - 1]]] = 1.0;
                    inputs.Add(new Tensor(inputData, new int[] { charToIndex.Count }));

                    // Alvo: codifica one-hot o próximo caractere
                    double[] targetData = new double[charToIndex.Count];
                    targetData[charToIndex[nextChar]] = 1.0;
                    targets.Add(new Tensor(targetData, new int[] { charToIndex.Count }));
                }
            }

            return (inputs.ToArray(), targets.ToArray());
        }

        private Tensor TextToTensor(string text)
        {
            double[] inputData = new double[charToIndex.Count];
            if (text.Length > 0 && charToIndex.ContainsKey(text[text.Length - 1]))
            {
                inputData[charToIndex[text[text.Length - 1]]] = 1.0;
            }
            return new Tensor(inputData, new int[] { charToIndex.Count });
        }

        private Tensor CharToTensor(char c)
        {
            double[] inputData = new double[charToIndex.Count];
            inputData[charToIndex[c]] = 1.0;
            return new Tensor(inputData, new int[] { charToIndex.Count });
        }
    }
}
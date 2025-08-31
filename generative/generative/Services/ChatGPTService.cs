using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization; // Para serialização/desserialização JSON
using System.Threading.Tasks;
using Models;

namespace Services
{
    public class ChatGPTService
    {
        private readonly HttpClient _httpClient;
        private readonly string _openAIApiKey;
        private static readonly string endpoint = "https://api.openai.com/v1/chat/completions";
        private readonly string _chatGPTModel = "gpt-4o-mini"; // Atual

        public ChatGPTService(IConfiguration configuration, HttpClient httpClient)
        {
            _httpClient = httpClient;
            _openAIApiKey = configuration["ApiKeys:OpenAI"] ??
                            throw new ArgumentNullException(
                                "OpenAI API Key not configured in appsettings.json or environment variables (ApiKeys:OpenAI).");

            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _openAIApiKey);
            _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }

        /// <summary>
        /// Consulta a API do ChatGPT para obter informações ou um resumo sobre um tópico/texto
        /// através de chamadas HTTP diretas.
        /// </summary>
        /// <param name="prompt">O tópico ou texto para o ChatGPT processar (e.g., "Resuma sobre...", "Forneça informações sobre...").</param>
        /// <param name="maxTokens">Número máximo de tokens para a resposta do ChatGPT.</param>
        /// <param name="temperature">Controla a aleatoriedade da resposta. 0.0 para mais determinístico, 1.0 para mais criativo.</param>
        /// <returns>Uma lista de strings com a informação sintetizada.</returns>
        public async Task<List<string>> GetInformationFromChatGPT(string prompt, int maxTokens = 500,
            float temperature = 0.7f)
        {
            Console.WriteLine(
                $"ChatGPTService: Consultando ChatGPT (via HttpClient) com prompt: '{prompt.Substring(0, Math.Min(100, prompt.Length))}...'");
            List<string> responseContent = new();

            try
            {
                var requestBody = new ChatCompletionRequest
                {
                    Model = _chatGPTModel,
                    Messages = new List<ChatMessage>
                    {
                        new ChatMessage
                        {
                            Role = "system",
                            Content =
                                "You are a helpful and concise assistant. Provide factual and well-structured information."
                        },
                        new ChatMessage { Role = "user", Content = prompt }
                    },
                    MaxTokens = maxTokens,
                    Temperature = temperature
                };

                var jsonRequest = JsonSerializer.Serialize(requestBody,
                    new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
                var content = new StringContent(jsonRequest, Encoding.UTF8, "application/json");

                HttpResponseMessage httpResponse = await _httpClient.PostAsync(endpoint, content);
                httpResponse.EnsureSuccessStatusCode();

                string jsonResponse = await httpResponse.Content.ReadAsStringAsync();
                var completionResult = JsonSerializer.Deserialize<ChatCompletionResponse>(jsonResponse);

                if (completionResult?.Choices?.Any() == true)
                {
                    var fullContent = completionResult.Choices.First().Message?.Content;
                    if (!string.IsNullOrEmpty(fullContent))
                    {
                        Console.WriteLine(
                            $"ChatGPTService: Resposta obtida (parcial):\n{fullContent.Substring(0, Math.Min(200, fullContent.Length))}...");
                        responseContent = fullContent
                            .Split(new[] { "\n\n", "\n" }, StringSplitOptions.RemoveEmptyEntries)
                            .ToList();
                    }
                }
                else
                {
                    Console.WriteLine(
                        "ChatGPTService: Consulta ao ChatGPT não retornou uma resposta válida ou ocorreu um erro.");
                }
            }
            catch (HttpRequestException httpEx)
            {
                Console.WriteLine($"ChatGPTService: Erro HTTP ao consultar ChatGPT: {httpEx.Message}");
            }
            catch (JsonException jsonEx)
            {
                Console.WriteLine(
                    $"ChatGPTService: Erro de desserialização JSON da resposta do ChatGPT: {jsonEx.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ChatGPTService: Exceção inesperada ao consultar ChatGPT: {ex.Message}");
            }

            return responseContent;
        }
    }
}
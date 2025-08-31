using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization; // Para atributos JsonPropertyName
using System.Threading.Tasks;
using Models; // Assumindo que as novas classes de modelo estão neste namespace

namespace Services
{
    public class GeminiService
    {
        private readonly HttpClient _httpClient;
        private readonly string _geminiApiKey;
        private static readonly string _baseEndpoint = "https://generativelanguage.googleapis.com/v1beta/models/";
        private readonly string _geminiModel = "gemini-2.0-flash"; // Modelo moderno e eficiente

        public GeminiService(IConfiguration configuration, HttpClient httpClient)
        {
            _httpClient = httpClient;
            // Busca a chave da API do Gemini da configuração
            _geminiApiKey = configuration["ApiKeys:Gemini"] ??
                            throw new ArgumentNullException(
                                "Gemini API Key not configured in appsettings.json or environment variables (ApiKeys:Gemini).");

            // Limpa cabeçalhos de autorização anteriores, pois a chave será enviada na URL
            _httpClient.DefaultRequestHeaders.Clear();
            _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }

        /// <summary>
        /// Consulta a API do Google Gemini para obter informações ou um resumo sobre um tópico/texto.
        /// </summary>
        /// <param name="prompt">O tópico ou texto para o Gemini processar.</param>
        /// <param name="maxTokens">Número máximo de tokens para a resposta do Gemini.</param>
        /// <param name="temperature">Controla a aleatoriedade da resposta. 0.0 para mais determinístico, 1.0 para mais criativo.</param>
        /// <returns>Uma lista de strings com a informação sintetizada.</returns>
        public async Task<List<string>> GetInformationFromGemini(string prompt, int maxTokens = 500,
            float temperature = 0.7f)
        {
            Console.WriteLine(
                $"GeminiService: Consultando Gemini com prompt: '{prompt.Substring(0, Math.Min(100, prompt.Length))}...'");
            List<string> responseContent = new();

            // Constrói a URL completa com o modelo e a chave da API
            var endpoint = $"{_baseEndpoint}{_geminiModel}:generateContent?key={_geminiApiKey}";

            try
            {
                // A API do Gemini não tem uma role "system" separada como o OpenAI.
                // A instrução do sistema pode ser incluída no início do prompt do usuário.
                var fullPrompt = "Você é um assistente prestativo e conciso. Forneça informações factuais e bem estruturadas. Responda sempre em português do Brasil.\\n\\n" + prompt;

                var requestBody = new GeminiRequest
                {
                    Contents = new List<Content>
                    {
                        new Content
                        {
                            Parts = new List<Part>
                            {
                                new Part { Text = fullPrompt }
                            }
                        }
                    },
                    GenerationConfig = new GenerationConfig
                    {
                        Temperature = temperature,
                        MaxOutputTokens = maxTokens
                    }
                };

                var jsonRequest = JsonSerializer.Serialize(requestBody,
                    new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
                var content = new StringContent(jsonRequest, Encoding.UTF8, "application/json");

                HttpResponseMessage httpResponse = await _httpClient.PostAsync(endpoint, content);
                httpResponse.EnsureSuccessStatusCode();

                string jsonResponse = await httpResponse.Content.ReadAsStringAsync();
                var geminiResponse = JsonSerializer.Deserialize<GeminiResponse>(jsonResponse);
                
                // Extrai o texto da estrutura de resposta do Gemini
                var fullContent = geminiResponse?.Candidates?.FirstOrDefault()?.Content?.Parts?.FirstOrDefault()?.Text;
                if (!string.IsNullOrEmpty(fullContent))
                {
                    Console.WriteLine(
                        $"GeminiService: Resposta obtida (parcial):\n{fullContent.Substring(0, Math.Min(200, fullContent.Length))}...");
                    responseContent = fullContent
                        .Split(new[] { "\n\n", "\n" }, StringSplitOptions.RemoveEmptyEntries)
                        .ToList();
                }
                else
                {
                    Console.WriteLine(
                        "GeminiService: Consulta ao Gemini não retornou uma resposta válida ou ocorreu um erro.");
                }
            }
            catch (HttpRequestException httpEx)
            {
                Console.WriteLine($"GeminiService: Erro HTTP ao consultar Gemini: {httpEx.Message}");
                // Opcional: Logar o corpo da resposta em caso de erro para depuração
                if (httpEx.StatusCode.HasValue)
                {
                    Console.WriteLine($"Status Code: {httpEx.StatusCode}");
                    if(httpEx.InnerException?.Source != null)
                    {
                        // Se a resposta contiver detalhes do erro, eles podem estar aqui
                        // string errorBody = await httpEx.Response.Content.ReadAsStringAsync();
                        // Console.WriteLine($"Error Body: {errorBody}");
                    }
                }
            }
            catch (JsonException jsonEx)
            {
                Console.WriteLine(
                    $"GeminiService: Erro de desserialização JSON da resposta do Gemini: {jsonEx.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GeminiService: Exceção inesperada ao consultar Gemini: {ex.Message}");
            }

            return responseContent;
        }
    }
}
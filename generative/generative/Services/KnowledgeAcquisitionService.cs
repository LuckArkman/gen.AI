// Services/KnowledgeAcquisitionService.cs
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Services
{
    public class KnowledgeAcquisitionService
    {
        private readonly ChatGPTService _chatGPTService;
        private readonly GoogleSearchService _googleSearchService;
        private readonly GeminiService _geminiService;
        private readonly TextProcessorService _textProcessorService;
        private readonly ILogger<KnowledgeAcquisitionService> _logger; // Para logs

        public KnowledgeAcquisitionService(ChatGPTService chatGPTService, 
            GoogleSearchService googleSearchService,
            GeminiService geminiService,
            TextProcessorService textProcessorService,
            ILogger<KnowledgeAcquisitionService> logger) // Injetar logger
        {
            _chatGPTService = chatGPTService;
            _googleSearchService = googleSearchService;
            _geminiService = geminiService;
            _textProcessorService = textProcessorService;
            _logger = logger; // Atribuir logger
        }

        /// <summary>
        /// Busca informações sobre um tópico, priorizando ChatGPT e usando Google como fallback.
        /// Retorna uma lista de strings com informações relevantes e o nome da fonte.
        /// </summary>
        /// <param name="query">O tópico da busca.</param>
        /// <returns>Tupla: (Lista de strings com conteúdo, Nome da fonte)</returns>
        public async Task<(List<string> content, string sourceName)> GetInformation(string query)
        {
            List<string> content = new List<string>();
            string sourceName = "Nenhuma";

            // 1. Tentar ChatGPT (fonte primária)
            Console.WriteLine($"KnowledgeAcquisition: Tentando fonte primária (ChatGPT) para '{query}'...");
            content = await _geminiService.GetInformationFromGemini(query);
            if (content.Any() && !string.IsNullOrWhiteSpace(string.Join("", content)))
            {
                sourceName = "ChatGPT API";
                Console.WriteLine("KnowledgeAcquisition: Informação obtida do ChatGPT.");
                return (content, sourceName);
            }

            Console.WriteLine("KnowledgeAcquisition: ChatGPT não retornou informações relevantes. Tentando fonte secundária (Google).");

            // 2. Tentar Google Search (fonte secundária)
            // Se o conteúdo do Google for bruto (snippets e texto de páginas),
            // podemos passá-lo para o ChatGPT para um resumo mais inteligente,
            // ou apenas resumir localmente se a chave do ChatGPT for limitada/indisponível.
            
            List<string> googleContent = await _googleSearchService.SearchAndExtractText(query);
            if (googleContent.Any() && !string.IsNullOrWhiteSpace(string.Join("", googleContent)))
            {
                // Opcional: Passar o conteúdo do Google pelo ChatGPT para um resumo inteligente
                // Se a intenção é que o ChatGPT sempre processe a info para resumo.
                // Isso incorre em custo adicional do ChatGPT.
                Console.WriteLine("KnowledgeAcquisition: Informação bruta obtida do Google. Opcionalmente, processando com ChatGPT para síntese.");
                var synthesizedContent = await _geminiService.GetInformationFromGemini($"Summarize and synthesize the following search results about '{query}':\n\n{string.Join("\n\n", googleContent)}", 1000);

                if (synthesizedContent.Any() && !string.IsNullOrWhiteSpace(string.Join("", synthesizedContent)))
                {
                    content = synthesizedContent;
                    sourceName = "Google Search (via ChatGPT Synthesis)";
                    Console.WriteLine("KnowledgeAcquisition: Informação do Google sintetizada pelo ChatGPT.");
                }
                else
                {
                    // Fallback se ChatGPT falhar na síntese
                    content = googleContent;
                    sourceName = "Google Search (raw)";
                    Console.WriteLine("KnowledgeAcquisition: Informação do Google usada diretamente.");
                }
                return (content, sourceName);
            }

            Console.WriteLine("KnowledgeAcquisition: Nenhuma fonte externa retornou informações relevantes.");
            return (new List<string>(), "Nenhuma");
        }
        public async Task<List<string>> GetSummarizationFromExternalService(string textToSummarize, int maxTokens = 500)
        {
            _logger.LogInformation("KnowledgeAcquisition: Solicitando resumo externo de texto com {MaxTokens} tokens...", maxTokens);
            string prompt = $"Please provide a concise and factual summary of the following text:\n\n{textToSummarize}";
            return await _geminiService.GetInformationFromGemini(prompt, maxTokens);
        }
    }
}
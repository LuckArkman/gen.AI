// Services/WebSearchService.cs
using System.Net.Http;
using System.Threading.Tasks;
using HtmlAgilityPack; // Para parsing HTML

namespace Services
{
    public class WebSearchService
    {
        private readonly HttpClient _httpClient;
        private readonly string _searchApiKey; // Configurar via IConfiguration

        public WebSearchService(HttpClient httpClient, IConfiguration configuration)
        {
            _httpClient = httpClient;
            _searchApiKey = configuration["ApiKeys:GoogleSearch"] ?? throw new ArgumentNullException("GoogleSearch API Key not configured.");
        }

        public async Task<List<string>> SearchAndExtractText(string query, int numResults = 3)
        {
            Console.WriteLine($"Simulando busca na internet para: '{query}'");
            var simulatedResults = new List<string>
            {
                $"Informação muito relevante sobre '{query}' encontrada na web. A data de hoje é {DateTime.Now:yyyy-MM-dd HH:mm:ss}.",
                $"Um artigo recente sobre '{query}' aponta novas descobertas...",
                $"Dados importantes sobre '{query}' de uma fonte confiável: o mundo está evoluindo!"
            };

            // Exemplo simplificado de extração HTML (se tivéssemos URLs reais)
            // string htmlContent = await _httpClient.GetStringAsync(url);
            // var doc = new HtmlDocument();
            // doc.LoadHtml(htmlContent);
            // var textNodes = doc.DocumentNode.SelectNodes("//body//text()");
            // return textNodes?.Select(n => n.InnerText).Where(s => !string.IsNullOrWhiteSpace(s)).ToList() ?? new List<string>();

            return simulatedResults.Take(numResults).ToList();
        }
    }
}
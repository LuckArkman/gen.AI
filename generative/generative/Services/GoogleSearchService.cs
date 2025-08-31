using HtmlAgilityPack;
using Newtonsoft.Json;

namespace Services
{
    public class GoogleSearchService // Renomeado de WebSearchService para mais clareza
    {
        private readonly HttpClient _httpClient;
        private readonly string _apiKey;
        private readonly string _cx; // Custom Search Engine ID
        private readonly TextProcessorService _textProcessorService;

        public GoogleSearchService(HttpClient httpClient, IConfiguration configuration,TextProcessorService textProcessorService)
        {
            _textProcessorService = textProcessorService;
            _httpClient = httpClient;
            _apiKey = configuration["ApiKeys:GoogleSearch"] ?? 
                      throw new ArgumentNullException("GoogleSearch API Key not configured.");
            _cx = configuration["ApiKeys:GoogleSearchCx"] ?? 
                  throw new ArgumentNullException("GoogleSearch CX (Custom Search Engine ID) not configured.");
        }

        public async Task<List<string>> SearchAndExtractText(string query, int numResults = 3)
        {
            Console.WriteLine($"Consultando Google Search API para: '{query}'...");
            var results = new List<string>();
            try
            {
                // Exemplo de URL para Google Custom Search JSON API
                // Substitua pelo seu endpoint real da API e parâmetros
                var apiUrl = $"https://www.googleapis.com/customsearch/v1?key={_apiKey}&cx={_cx}&q={Uri.EscapeDataString(query)}&num={numResults}";
                var response = await _httpClient.GetStringAsync(apiUrl);
                
                // Parse a resposta JSON
                // Pode ser necessário instalar Newtonsoft.Json: dotnet add package Newtonsoft.Json
                dynamic? searchData = JsonConvert.DeserializeObject(response);

                if (searchData?.items != null)
                {
                    foreach (var item in searchData.items)
                    {
                        string? snippet = item.snippet; // Descrição curta do resultado
                        string? link = item.link; // URL do resultado

                        if (!string.IsNullOrEmpty(snippet))
                        {
                            results.Add(snippet);
                        }
                        
                        // Opcional: visitar o link e extrair mais texto
                        // Isso é mais custoso e pode ser lento
                        if (!string.IsNullOrEmpty(link) && Uri.IsWellFormedUriString(link, UriKind.Absolute))
                        {
                            try
                            {
                                // Apenas para links HTTP/HTTPS
                                if (link.StartsWith("http://") || link.StartsWith("https://"))
                                {
                                    Console.WriteLine($"Visitando link: {link}");
                                    // Adicionar um limite de tempo ou tamanho para evitar downloads grandes
                                    var pageContent = await _httpClient.GetStringAsync(link);
                                    var htmlDoc = new HtmlDocument();
                                    htmlDoc.LoadHtml(pageContent);
                                    // Extrair texto limpo, talvez apenas de certas tags (p, div etc.)
                                    var textNodes = htmlDoc.DocumentNode.SelectNodes("//body//p|//body//h1|//body//h2|//body//h3|//body//li");
                                    if (textNodes != null)
                                    {
                                        var extractedText = string.Join("\n", textNodes.Select(n => n.InnerText.Trim()).Where(s => !string.IsNullOrEmpty(s)));
                                        // Limitar o tamanho do texto extraído para não sobrecarregar
                                        results.Add(_textProcessorService.Summarize(extractedText, 200)); // Usar summarize para pegar um trecho
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Erro ao extrair conteúdo do link {link}: {ex.Message}");
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro ao consultar Google Search API: {ex.Message}");
            }
            return results;
        }
    }
}
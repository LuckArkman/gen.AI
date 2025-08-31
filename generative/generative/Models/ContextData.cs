namespace Models;

public class ContextData
{
    public string TopicHash { get; set; } // Identificador do tópico
    public string Summary { get; set; }   // Resumo do contexto
    public List<string> RelatedUrls { get; set; } // URLs das fontes
    public long ExternalLastUpdatedTicks { get; set; } // Timestamp da última atualização externa (internet)
}
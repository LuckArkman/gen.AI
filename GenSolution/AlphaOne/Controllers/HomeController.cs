using System.Diagnostics;
using System.Security.Claims;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using AlphaOne.Models;
using AlphaOne.Services;
using Microsoft.AspNetCore.Authorization;

namespace AlphaOne.Controllers;

//[Authorize]
public class HomeController : Controller
{
    private readonly ConversationService _conversationService;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly string _apiBaseUrl;
    private readonly IConfiguration _configuration;
    private readonly ILogger<HomeController> _logger; // <-- Adicionar esta declaração de campo

    public HomeController(ConversationService conversationService, 
        IHttpClientFactory httpClientFactory,
        IConfiguration configuration,
        ILogger<HomeController> logger) // <-- Injetar ILogger aqui
    {
        _conversationService = conversationService;
        _httpClientFactory = httpClientFactory;
        _configuration = configuration;
        _apiBaseUrl = _configuration["GenerativeAIService:ApiBaseUrl"] ?? 
                      throw new ArgumentNullException("GenerativeAIService:ApiBaseUrl not configured.");
        _logger = logger; // <-- Atribuir o logger injetado ao campo
    }

    [HttpGet]
    public IActionResult Index()
    {
        ViewBag.ApiBaseUrl = _apiBaseUrl;
        ViewBag.BodyClass = "chat-layout";
        return View();
    }

    // --- Endpoints para Gerenciamento de Conversas (chamados via AJAX do frontend) ---
    [HttpGet("api/chat/GetConversations")]
    public async Task<IActionResult> GetConversations()
    {
        var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (userId == null) return Unauthorized();

        var conversations = await _conversationService.GetConversationsByUserIdAsync(userId);
        // Retorna apenas o ID e o Título para a barra lateral
        return Ok(conversations.Select(c => new { c.Id, c.Title }).ToList());
    }

    [HttpGet("api/chat/GetConversation/{conversationId}")]
    public async Task<IActionResult> GetConversation(string conversationId)
    {
        var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (userId == null) return Unauthorized();

        var conversation = await _conversationService.GetConversationByIdAsync(conversationId, userId);
        if (conversation == null) return NotFound();

        return Ok(conversation);
    }

    [HttpPost("api/chat/GenerateAndSave")]
    public async Task<IActionResult> GenerateAndSave([FromBody] GenerateRequestModel request)
    {
        var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (userId == null) return Unauthorized();

        string userMessage = request.SeedText ?? string.Empty;
        string aiResponse = string.Empty;

        try
        {
            var httpClient = _httpClientFactory.CreateClient();
            var jsonContent = new StringContent(
                JsonSerializer.Serialize(request),
                Encoding.UTF8,
                "application/json"
            );

            // Chama a API GenerativeAI externa
            var apiResponse = await httpClient.PostAsync($"{_apiBaseUrl}/generate", jsonContent);
            apiResponse.EnsureSuccessStatusCode(); // Lança exceção para códigos de erro HTTP
            aiResponse = await apiResponse.Content.ReadAsStringAsync();

            // Salva a conversa no MongoDB
            await _conversationService.AddMessageToConversationAsync(userId, request.ConversationId, userMessage,
                aiResponse);

            // Retorna a resposta da AI e talvez o ID da conversa (se for nova)
            return Ok(new { AiResponse = aiResponse, ConversationId = request.ConversationId });
        }
        catch (HttpRequestException ex)
        {
            _logger.LogError(ex, "Erro ao chamar a API GenerativeAI: {Message}", ex.Message);
            return StatusCode(500, new { Error = $"Erro ao gerar resposta: {ex.Message}" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro inesperado em GenerateAndSave: {Message}", ex.Message);
            return StatusCode(500, new { Error = $"Erro interno: {ex.Message}" });
        }
    }
}
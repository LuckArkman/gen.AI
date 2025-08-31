// GenerativeAIWebApp/Middlewares/DevBypassAuthMiddleware.cs (NOVO ARQUIVO)
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting; // Para IWebHostEnvironment
using System.Threading.Tasks;

namespace AlphaOne.Middlewares
{
    public class DevBypassAuthMiddleware
    {
        private readonly RequestDelegate _next;
        private readonly IWebHostEnvironment _env;

        public DevBypassAuthMiddleware(RequestDelegate next, IWebHostEnvironment env)
        {
            _next = next;
            _env = env;
        }

        public async Task InvokeAsync(HttpContext context)
        {
            if (_env.IsDevelopment())
            {
                // Verifica se a requisição é para a Home/Index
                if (context.Request.Path.Value != null && 
                    (context.Request.Path.Value.Equals("/", StringComparison.OrdinalIgnoreCase) ||
                     context.Request.Path.Value.Equals("/Home", StringComparison.OrdinalIgnoreCase) ||
                     context.Request.Path.Value.Equals("/Home/Index", StringComparison.OrdinalIgnoreCase)))
                {
                    // Se o usuário não está autenticado, mas está em desenvolvimento e indo para Home/Index
                    if (!context.User.Identity?.IsAuthenticated ?? false)
                    {
                        // Cria um Identity de mock (temporário)
                        var claims = new System.Security.Claims.Claim[]
                        {
                            new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.NameIdentifier, "dev_user_id"),
                            new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.Name, "dev_user"),
                            new System.Security.Claims.Claim(System.Security.Claims.ClaimTypes.Email, "dev@example.com")
                        };
                        var identity = new System.Security.Claims.ClaimsIdentity(claims, "DevBypassAuth");
                        context.User = new System.Security.Claims.ClaimsPrincipal(identity);
                        Console.WriteLine("DevBypassAuthMiddleware: Automatically authenticated as dev user for Home/Index.");
                    }
                }
            }
            await _next(context); // Continua para o próximo middleware
        }
    }
}
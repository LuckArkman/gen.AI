// Importações necessárias
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using AlphaOne.Data; // Seu namespace de dados, certifique-se que está correto
using AlphaOne.Services; // Seu namespace de serviços, certifique-se que está correto
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.Extensions.DependencyInjection; // Para a extensão AddHttpClient
using Microsoft.AspNetCore.Http; // Para IHttpContextAccessor
using AlphaOne.Middlewares; // Seu namespace para o middleware de bypass

var builder = WebApplication.CreateBuilder(args);

// --- Configuração dos Serviços (equivalente ao ConfigureServices em Startup.cs) ---

// 1. Configuração do Contexto do Banco de Dados PostgreSQL
var pgConnectionString = builder.Configuration.GetConnectionString("PostgreSQLConnection") ??
                       throw new InvalidOperationException("Connection string 'PostgreSQLConnection' not found in appsettings.json.");
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseNpgsql(pgConnectionString)); // Usando Npgsql para PostgreSQL

// 2. Configuração do ASP.NET Core Identity e Autenticação
builder.Services.AddIdentity<ApplicationUser, IdentityRole>(options => // Usando ApplicationUser e IdentityRole
    {
        options.SignIn.RequireConfirmedAccount = false; // Conforme requisito
        // Opções de senha relaxadas conforme requisito
        options.Password.RequireDigit = false;
        options.Password.RequiredLength = 6;
        options.Password.RequireNonAlphanumeric = false;
        options.Password.RequireUppercase = false;
        options.Password.RequireLowercase = false;
    })
    .AddEntityFrameworkStores<ApplicationDbContext>()
    .AddDefaultTokenProviders(); // Necessário para reset de senha, etc.

// Configuração do Cookie de Autenticação para redirecionamento
builder.Services.ConfigureApplicationCookie(options =>
{
    options.LoginPath = "/Account/Login"; // Página de login
    options.AccessDeniedPath = "/Account/AccessDenied"; // Página de acesso negado (opcional)
    options.LogoutPath = "/Account/Logout"; // Página de logout
});

// 3. Configuração dos Controladores e Views
builder.Services.AddControllersWithViews();

// Adiciona um filtro para exceções do banco de dados em páginas de desenvolvimento (útil para migrações)
builder.Services.AddDatabaseDeveloperPageExceptionFilter();

// 4. Adição de Serviços Customizados
// Serviços de backup e conversas MongoDB
builder.Services.AddSingleton<MongoBackupService>();
builder.Services.AddSingleton<ConversationService>();

// Para chamadas HTTP à API generativa externa (IHttpClientFactory)
builder.Services.AddHttpClient();

// Para acessar HttpContext e UserId em serviços (necessário para Identity)
builder.Services.AddHttpContextAccessor();

// Não é necessário adicionar ILogger<T> explicitamente; WebApplication.CreateBuilder já configura o logging padrão.

var app = builder.Build();

// --- Configuração do Pipeline de Requisições HTTP (equivalente ao Configure em Startup.cs) ---

// 1. Tratamento de Exceções
if (app.Environment.IsDevelopment())
{
    app.UseMigrationsEndPoint(); // Para aplicar migrações do EF Core automaticamente em dev
    app.UseDeveloperExceptionPage(); // Página de erro detalhada em desenvolvimento
}
else
{
    app.UseExceptionHandler("/Home/Error"); // Página de erro genérica em produção
    app.UseHsts(); // Proteção HSTS em produção
}

// 2. Redirecionamento HTTPS e Arquivos Estáticos
app.UseHttpsRedirection();
app.UseStaticFiles();

// 3. Roteamento
app.UseRouting();

// 4. Middleware Customizado para Bypass de Autenticação em Desenvolvimento
// Este middleware deve vir ANTES de UseAuthentication e UseAuthorization
app.UseMiddleware<DevBypassAuthMiddleware>();

// 5. Autenticação e Autorização (ORDEM IMPORTA!)
app.UseAuthentication(); // Deve ser chamado antes de UseAuthorization
app.UseAuthorization(); // Deve ser chamado após UseAuthentication

// 6. Mapeamento de Endpoints
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Account}/{action=Login}/{id?}"); // Rota padrão inicial (Login)

// Rota para a interface do chat (Home), acessível apenas após autenticação
app.MapControllerRoute(
    name: "chat",
    pattern: "Home/{action=Index}/{id?}", // Ex: /Home/Index
    defaults: new { controller = "Home", action = "Index" } // Redireciona para Home/Index
);

app.Run();
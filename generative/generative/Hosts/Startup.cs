// GenerativeAIAPI/Startup.cs (ou Hosts/Startup.cs se for o nome do seu projeto)

using Microsoft.AspNetCore.Builder; // Para IApplicationBuilder
using Microsoft.AspNetCore.Hosting; // Para IWebHostEnvironment
using Microsoft.Extensions.Configuration; // Para IConfiguration
using Microsoft.Extensions.DependencyInjection; // Para IServiceCollection, AddSingleton, AddHttpClient
using Microsoft.Extensions.Hosting; // Para IsDevelopment
using System; // Para ArgumentNullException
using System.IO; // Para Path
using System.Linq; // Para FirstOrDefault (se Cloo estiver envolvido na startup)

// Namespaces dos seus próprios serviços, modelos e libs
using Core; // Se NeuralNetwork ou outros tipos Core forem usados na startup
using Models; // Se modelos como ContextInfo forem usados na startup
using BinaryTreeSwapFile; // Para BinaryTreeFileStorage
using Services; // <-- O namespace correto para seus serviços customizados

namespace Hosts // ou o namespace correto para a classe Startup da API
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        public void ConfigureServices(IServiceCollection services)
        {
            // --- Configurações da API Básicas (Sem Duplicações) ---
            services.AddControllers();
            services.AddEndpointsApiExplorer(); // Para Swagger
            services.AddSwaggerGen();           // Para Swagger UI

            // Habilita o CORS
            services.AddCors(options =>
            {
                options.AddPolicy("AllowAll", builder =>
                {
                    builder.AllowAnyOrigin()
                        .AllowAnyMethod()
                        .AllowAnyHeader();
                });
            });

            // --- Registro dos Serviços de IA e Memória (Sem Duplicações) ---
            
            // TextProcessorService (depende apenas da própria lógica)
            services.AddSingleton<TextProcessorService>();

            // ChatGPTService (usa HttpClient, então usa AddHttpClient<T>)
            services.AddHttpClient<ChatGPTService>(); 
            services.AddHttpClient<GeminiService>(); 
            
            // GoogleSearchService (usa HttpClient, então usa AddHttpClient<T>)
            services.AddHttpClient<GoogleSearchService>(); 
            
            // KnowledgeAcquisitionService (orquestra ChatGPTService e GoogleSearchService)
            services.AddSingleton<KnowledgeAcquisitionService>(); 
            
            // ContextManager (depende de TextProcessorService, KnowledgeAcquisitionService e BinaryTreeFileStorage)
            services.AddSingleton<ContextManager>(); 
            
            // BinaryTreeFileStorage (instanciado com factory para pegar o caminho da config)
            services.AddSingleton(provider => 
            {
                var config = provider.GetRequiredService<IConfiguration>();
                // modelDir precisa ser obtido para Path.Combine
                var modelDir = config["ModelSettings:ModelDirectory"] ?? "/home/mplopes/Documentos/generative/generative/";
                var memoryFilePath = config["ModelSettings:MemoryFilePath"] ?? Path.Combine(modelDir, "AIModelMem.dat");
                
                var storage = new BinaryTreeFileStorage(memoryFilePath);
                // NOTA: A lógica de GenerateEmptyTree() para _memoryStorage é agora chamada no construtor do GenerativeAIController
                // para garantir que seja executada apenas uma vez na inicialização do controlador da API.
                return storage;
            });
            
            // Serviço de Logging padrão do .NET Core
            services.AddLogging();
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
                // Habilita o Swagger e o Swagger UI em desenvolvimento
                app.UseSwagger();
                app.UseSwaggerUI();
            }
            // Não há mais else para UseExceptionHandler/HSTS porque é uma API
            // e os erros são retornados via IActionResult.

            app.UseRouting();
            app.UseCors("AllowAll"); // Deve vir depois de UseRouting e antes de UseEndpoints
            
            // Autenticação e Autorização são para a UI, não para a API GENERATIVA em si.
            // Se a API tiver endpoints que precisam de autenticação, adicione aqui:
            // app.UseAuthentication();
            // app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers(); // Mapeia todos os controladores da API
            });
        }
    }
}
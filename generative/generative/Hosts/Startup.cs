using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;
using Core;
using BinaryTreeSwapFile;
using Services;
using GenerativeAIAPI.Controllers;

namespace Hosts
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
            services.AddControllers();
            services.AddEndpointsApiExplorer();
            services.AddSwaggerGen();
            services.AddCors(options =>
            {
                options.AddPolicy("AllowAll", builder =>
                {
                    builder.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader();
                });
            });

            // Registro dos Serviços
            services.AddSingleton<ListenerService>();
            services.AddSingleton<TextProcessorService>();
            services.AddHttpClient<GeminiService>();
            services.AddHttpClient<GoogleSearchService>(); 
            services.AddSingleton<KnowledgeAcquisitionService>(); 
            services.AddSingleton<DatasetService>();
            services.AddSingleton<ContextManager>();
            services.AddSingleton<GenerativeAIController>();
            
            
            services.AddSingleton(provider => 
            {
                var config = provider.GetRequiredService<IConfiguration>();
                var modelDir = config["ModelSettings:ModelDirectory"] ?? "/home/mplopes/Documentos/generative/generative/";
                var memoryFilePath = config["ModelSettings:MemoryFilePath"] ?? Path.Combine(modelDir, "AIModelMem.dat");
                return new BinaryTreeFileStorage(memoryFilePath);
            });
            
            // CORREÇÃO: O Controller é um singleton porque detém o estado do modelo.
            // O Listener é um serviço de background que o encontrará via IServiceProvider.
            
            services.AddLogging();
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseRouting();
            app.UseCors("AllowAll");
            
            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
        }
    }
}
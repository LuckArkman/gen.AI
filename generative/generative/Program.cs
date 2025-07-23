using Core;
using Hosts;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;
using Microsoft.Extensions.Configuration; 

public class Program
{
    public static void Main(string[] args)
    {
        var host = CreateHostBuilder(args).Build();
        
        var configuration = host.Services.GetRequiredService<IConfiguration>();
        string modelDir = "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/";

        if (args.Length > 0 && args[0].Equals("--train", StringComparison.OrdinalIgnoreCase))
        {
            int totalEpochs = 100; // Padrão
            int startEpoch = 1;   // Padrão
            int contextWindowSize = 10; // Padrão

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i].Equals("--epoch", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    if (int.TryParse(args[i+1], out int value))
                    {
                        totalEpochs = Math.Max(1, value);
                    }
                }
                else if (args[i].Equals("--start-epoch", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    if (int.TryParse(args[i+1], out int value))
                    {
                        startEpoch = Math.Max(1, value);
                    }
                }
                // Novo argumento para ContextWindowSize
                else if (args[i].Equals("--window-size", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    if (int.TryParse(args[i+1], out int value))
                    {
                        contextWindowSize = Math.Max(1, value);
                    }
                }
            }

            if (startEpoch > totalEpochs)
            {
                Console.WriteLine($"Aviso: startEpoch ({startEpoch}) é maior que totalEpochs ({totalEpochs}). Ajustando startEpoch para 1.");
                startEpoch = 1;
            }

            RunTraining(startEpoch, totalEpochs, contextWindowSize, modelDir); // Passa contextWindowSize
        }
        else
        {
            host.Run();
        }
    }

    private static void RunTraining(int startEpoch, int totalEpochs, int contextWindowSize, string modelDir) // Adicionado contextWindowSize
    {
        try
        {
            Console.WriteLine($"Iniciando modo de treinamento (época inicial: {startEpoch}, total de épocas: {totalEpochs}, janela de contexto: {contextWindowSize})...");

            string datasetPath = "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/output/code";
            string modelPathTemplate = Path.Combine(modelDir, "model.json");
            string vocabPath = Path.Combine(modelDir, "vocab.txt");
            int hiddenSize = 256;
            double learningRate = 0.001;
            
            // Cria o diretório para os modelos, se não existir
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
                Console.WriteLine($"Diretório criado: {modelDir}");
            }

            // Inicializa o treinador, passando o ContextWindowSize
            var trainer = new Trainer(
                datasetPath: datasetPath,
                modelPathTemplate: modelPathTemplate,
                vocabPath: vocabPath,
                hiddenSize: hiddenSize,
                sequenceLength: contextWindowSize, // Passa contextWindowSize para sequenceLength do Trainer
                learningRate: learningRate,
                epochs: totalEpochs
            );

            trainer.Train(startEpoch);

            Console.WriteLine("Treinamento concluído com sucesso.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro durante o treinamento: {ex.Message}");
            Environment.Exit(1);
        }
    }

    public static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureAppConfiguration((hostingContext, config) =>
            {
                config.AddJsonFile("appsettings.json", optional: true, reloadOnChange: true);
                config.AddEnvironmentVariables();
                if (args != null)
                {
                    config.AddCommandLine(args);
                }
            })
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<Startup>();
            });
}
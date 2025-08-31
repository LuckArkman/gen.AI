using Core;
using BinaryTreeSwapFile;
using Services;
using Hosts;


public class Program
{
    public static void Main(string[] args)
    {
        // 1. Configurar o Host da Aplicação
        // CreateHostBuilder agora é para o modo Web.
        // Se isTrainingMode, configuramos um host mínimo para DI no modo train.
        
        bool isTrainingMode = args.Any(arg => arg.Equals("--train", StringComparison.OrdinalIgnoreCase));
        
        if (isTrainingMode)
        {
            Console.WriteLine("Modo de treinamento detectado.");
            // Constrói um host mínimo para resolver serviços necessários para o Trainer
            var builder = WebApplication.CreateBuilder(args);

            // Adiciona serviços mínimos para o Trainer
            builder.Services.AddSingleton<TextProcessorService>();
            builder.Services.AddSingleton(provider =>
            {
                var config = provider.GetRequiredService<IConfiguration>();
                var modelDir = config["ModelSettings:ModelDirectory"] ?? "/home/mplopes/Documentos/generative/generative/";
                var memoryFilePath = config["ModelSettings:MemoryFilePath"] ?? Path.Combine(modelDir, "AIModelMem.dat");
                
                var storage = new BinaryTreeFileStorage(memoryFilePath);
                // A lógica de GenerateEmptyTree() para _memoryStorage é agora chamada no construtor do GenerativeAIController
                // ou no Trainer, para garantir que seja executada apenas uma vez na inicialização.
                // No Trainer, já há uma verificação para isso.
                return storage;
            });
            builder.Services.AddLogging(); // Essencial para ILogger no Trainer

            var app = builder.Build(); // Constrói o app para ter um ServiceProvider

            // Cria um escopo de serviço para resolver as dependências para o Trainer
            using (var scope = app.Services.CreateScope())
            {
                var services = scope.ServiceProvider;
                var configuration = services.GetRequiredService<IConfiguration>();

                // Parâmetros de treinamento (lidos de appsettings e sobrescritos por CLI)
                int totalEpochs = configuration.GetValue<int>("TrainingSettings:Epochs", 100);
                int startEpoch = configuration.GetValue<int>("TrainingSettings:StartEpoch", 1);
                int contextWindowSize = configuration.GetValue<int>("ModelSettings:ContextWindowSize", 10);
                int chunkSize = configuration.GetValue<int>("TrainingSettings:ChunkSize", 1000); // Adicionado chunkSize

                // Sobrescreve com argumentos de linha de comando
                for (int i = 0; i < args.Length; i++)
                {
                    if (args[i].Equals("--epoch", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                    {
                        if (int.TryParse(args[i+1], out int value)) { totalEpochs = Math.Max(1, value); }
                    }
                    else if (args[i].Equals("--start-epoch", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                    {
                        if (int.TryParse(args[i+1], out int value)) { startEpoch = Math.Max(1, value); }
                    }
                    else if (args[i].Equals("--window-size", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                    {
                        if (int.TryParse(args[i+1], out int value)) { contextWindowSize = Math.Max(1, value); }
                    }
                    else if (args[i].Equals("--chunk-size", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length) // Adicionado parsing para chunkSize
                    {
                        if (int.TryParse(args[i+1], out int value)) { chunkSize = Math.Max(1, value); }
                    }
                }

                if (startEpoch > totalEpochs)
                {
                    Console.WriteLine($"Aviso: startEpoch ({startEpoch}) é maior que totalEpochs ({totalEpochs}). Ajustando startEpoch para 1.");
                    startEpoch = 1;
                }
                
                // Obtém as instâncias dos serviços para passar ao Trainer
                var textProcessorService = services.GetRequiredService<TextProcessorService>();
                var memoryStorage = services.GetRequiredService<BinaryTreeFileStorage>();
                
                RunTraining(startEpoch, totalEpochs, contextWindowSize, chunkSize, 
                            textProcessorService, memoryStorage, configuration);
            }
        }
        else
        {
            // --- Configuração para o modo de aplicação Web ---
            CreateWebHostBuilder(args).Build().Run();
        }
    }

    // Método auxiliar para executar o treinamento
    private static void RunTraining(int startEpoch, int totalEpochs, int contextWindowSize, int chunkSize, // Adicionado chunkSize
                                    TextProcessorService textProcessorService, BinaryTreeFileStorage memoryStorage, IConfiguration configuration) 
    {
        try
        {
            Console.WriteLine($"Iniciando modo de treinamento (época inicial: {startEpoch}, total de épocas: {totalEpochs}, janela de contexto: {contextWindowSize}, tamanho do chunk: {chunkSize})...");

            string datasetPath = configuration["TrainingSettings:DatasetPath"] ?? "/home/mplopes/Documentos/generative/generative/output/code";
            string modelDir = configuration["ModelSettings:ModelDirectory"] ?? "/home/mplopes/Documentos/generative/generative/"; // ModelDir lido da config
            string modelPathTemplate = Path.Combine(modelDir, "model.json"); 
            string vocabPath = Path.Combine(modelDir, "vocab.txt");
            int hiddenSize = configuration.GetValue<int>("ModelSettings:HiddenSize", 256);
            double learningRate = configuration.GetValue<double>("ModelSettings:LearningRate", 0.0001);
            
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
                Console.WriteLine($"Diretório criado: {modelDir}");
            }

            var trainer = new Trainer(
                datasetPath: datasetPath,
                modelPathTemplate: modelPathTemplate,
                vocabPath: vocabPath,
                hiddenSize: hiddenSize,
                sequenceLength: contextWindowSize,
                learningRate: learningRate,
                epochs: totalEpochs,
                textProcessorService: textProcessorService,
                memoryStorage: memoryStorage
            );

            trainer.Train(startEpoch, chunkSize); // CORRIGIDO: Passa chunkSize para o trainer.Train

            Console.WriteLine("Treinamento concluído com sucesso.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro durante o treinamento: {ex.Message}");
            Environment.Exit(1);
        }
    }

    // Método auxiliar para criar o Web Host (para o modo de aplicação Web)
    public static IHostBuilder CreateWebHostBuilder(string[] args) =>
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
                webBuilder.UseStartup<Startup>(); // Use a classe Startup
            });
}
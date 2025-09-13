using Core;
using BinaryTreeSwapFile;
using Services;
using Hosts;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

public class Program
{
    public static void Main(string[] args)
    {
        bool isTrainingMode = args.Any(arg => arg.Equals("--train", StringComparison.OrdinalIgnoreCase));
        
        if (isTrainingMode)
        {
            Console.WriteLine("Modo de treinamento detectado.");
            
            var host = Host.CreateDefaultBuilder(args)
                .ConfigureServices((context, services) =>
                {
                    services.AddSingleton<TextProcessorService>();
                    services.AddSingleton<DatasetService>();
                    services.AddSingleton<BinaryTreeFileStorage>(provider =>
                    {
                        var config = provider.GetRequiredService<IConfiguration>();
                        var modelDir = config["ModelSettings:ModelDirectory"] ?? "/home/mplopes/Documentos/generative/generative/";
                        var memoryFilePath = config["ModelSettings:MemoryFilePath"] ?? Path.Combine(modelDir, "AIModelMem.dat");
                        return new BinaryTreeFileStorage(memoryFilePath);
                    });
                })
                .Build();

            var services = host.Services;
            var configuration = services.GetRequiredService<IConfiguration>();
            var textProcessorService = services.GetRequiredService<TextProcessorService>();
            var memoryStorage = services.GetRequiredService<BinaryTreeFileStorage>();
            var datasetService = services.GetRequiredService<DatasetService>();

            // --- LÓGICA DE PARSING RESTAURADA ---
            // Parâmetros de treinamento (lidos de appsettings e sobrescritos por CLI)
            int totalEpochs = configuration.GetValue<int>("TrainingSettings:Epochs", 100);
            int startEpoch = configuration.GetValue<int>("TrainingSettings:StartEpoch", 1);
            int contextWindowSize = configuration.GetValue<int>("ModelSettings:ContextWindowSize", 10);
            int chunkSize = configuration.GetValue<int>("TrainingSettings:ChunkSize", 1000);

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
                else if (args[i].Equals("--chunk-size", StringComparison.OrdinalIgnoreCase) && i + 1 < args.Length)
                {
                    if (int.TryParse(args[i+1], out int value)) { chunkSize = Math.Max(1, value); }
                }
            }

            if (startEpoch > totalEpochs)
            {
                Console.WriteLine($"Aviso: startEpoch ({startEpoch}) é maior que totalEpochs ({totalEpochs}). Ajustando startEpoch para 1.");
                startEpoch = 1;
            }
            // --- FIM DA LÓGICA DE PARSING ---

            RunTraining(startEpoch, totalEpochs, contextWindowSize, chunkSize, 
                        textProcessorService, memoryStorage, datasetService, configuration);
        }
        else
        {
            // --- Configuração para o modo de aplicação Web ---
            CreateWebHostBuilder(args).Build().Run();
        }
    }

    // Método auxiliar para executar o treinamento
    private static void RunTraining(int startEpoch, int totalEpochs, int contextWindowSize, int chunkSize,
                                    TextProcessorService textProcessorService, 
                                    BinaryTreeFileStorage memoryStorage, 
                                    DatasetService datasetService, // Adiciona o novo serviço
                                    IConfiguration configuration) 
    {
        try
        {
            Console.WriteLine($"Iniciando modo de treinamento (época inicial: {startEpoch}, total de épocas: {totalEpochs}, janela de contexto: {contextWindowSize}, tamanho do chunk: {chunkSize})...");

            string datasetPath = configuration["TrainingSettings:DatasetPath"] ?? "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/output/code.txt";
            string modelDir = configuration["ModelSettings:ModelDirectory"] ?? "/home/mplopes/Documentos/GitHub/gen.AI/generative/generative/";
            string modelPathTemplate = Path.Combine(modelDir, "model.json"); 
            string vocabPath = Path.Combine(modelDir, "vocab.txt");
            int hiddenSize = configuration.GetValue<int>("ModelSettings:HiddenSize", 256);
            double learningRate = configuration.GetValue<double>("ModelSettings:LearningRate", 0.001);
            
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
            }

            var trainer = new Trainer(
                datasetPath: datasetPath,
                modelPathTemplate: modelPathTemplate,
                vocabPath: vocabPath,
                hiddenSize: hiddenSize,
                sequenceLength: contextWindowSize,
                initialLearningRate: learningRate,
                epochs: totalEpochs,
                textProcessorService: textProcessorService,
                memoryStorage: memoryStorage,
                datasetService: datasetService // Passa o novo serviço
            );

            trainer.Train(startEpoch, chunkSize);

            Console.WriteLine("Treinamento concluído com sucesso.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Erro durante o treinamento: {ex.Message}");
            Environment.Exit(1);
        }
    }

    public static IHostBuilder CreateWebHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<Startup>();
            });
}
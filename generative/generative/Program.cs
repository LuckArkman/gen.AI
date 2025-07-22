// Program.cs
using Core;
using Hosts;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;

public class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0].Equals("--train", StringComparison.OrdinalIgnoreCase))
        {
            int startEpoch = 1;
            if (args.Length > 2 && args[1].Equals("--epoch", StringComparison.OrdinalIgnoreCase) && int.TryParse(args[2], out int epoch))
            {
                startEpoch = Math.Max(1, epoch);
            }
            RunTraining(startEpoch);
        }
        else
        {
            CreateHostBuilder(args).Build().Run();
        }
    }

    private static void RunTraining(int startEpoch)
    {
        try
        {
            Console.WriteLine($"Iniciando modo de treinamento (época inicial: {startEpoch})...");

            // Configurações do treinamento
            string datasetPath = "/home/mplopes/RiderProjects/generative/generative/output/train_dataset.txt";
            string modelDir = "/home/mplopes/RiderProjects/generative/generative/models/";
            string modelPathTemplate = Path.Combine(modelDir, "model_epoch_10.json");
            string vocabPathTemplate = Path.Combine(modelDir, "vocab_epoch_10.txt");
            int hiddenSize = 256;
            int sequenceLength = 10;
            double learningRate = 0.01;
            int epochs = 10;

            // Criar o diretório para os modelos, se não existir
            if (!Directory.Exists(modelDir))
            {
                Directory.CreateDirectory(modelDir);
                Console.WriteLine($"Diretório criado: {modelDir}");
            }

            // Inicializa o treinador
            var trainer = new Trainer(
                datasetPath: datasetPath,
                modelPathTemplate: modelPathTemplate,
                vocabPathTemplate: vocabPathTemplate,
                hiddenSize: hiddenSize,
                sequenceLength: sequenceLength,
                learningRate: learningRate,
                epochs: epochs
            );

            // Verifica se existe um modelo treinado para a época inicial - 1
            if (startEpoch > 1)
            {
                string modelPath = modelPathTemplate.Replace("{epoch}", (startEpoch - 1).ToString());
                string vocabPath = vocabPathTemplate.Replace("{epoch}", (startEpoch - 1).ToString());
                if (File.Exists(modelPath) && File.Exists(vocabPath))
                {
                    Console.WriteLine($"Tentando carregar modelo e vocabulário da época {startEpoch - 1}...");
                    bool loaded = trainer.LoadModelAndVocabulary(modelPath, vocabPath);
                    if (!loaded)
                    {
                        Console.WriteLine("Falha ao carregar modelo ou vocabulário. Abortando.");
                        Environment.Exit(1);
                    }
                }
                else
                {
                    Console.WriteLine($"Modelo ou vocabulário da época {startEpoch - 1} não encontrado. Iniciando novo treinamento.");
                    startEpoch = 1; // Reseta para a primeira época
                }
            }

            // Executa o treinamento
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
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<Startup>();
            });
}
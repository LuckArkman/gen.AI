using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using BinaryTreeSwapFile;
using Core;
using Models;

namespace Services
{
    public class DatasetService
    {
        private static readonly ArrayPool<double> _arrayPool = ArrayPool<double>.Shared;

        // Processa o texto, armazena cada AMOSTRA individualmente, e retorna os offsets
        public List<long> PreprocessAndStoreSamples(
            string text,
            int contextWindowSize,
            Dictionary<string, int> tokenToIndex,
            string padToken,
            BinaryTreeFileStorage memoryStorage)
        {
            Console.WriteLine("Iniciando streaming e armazenamento de amostras para o bloco atual...");
            
            // 1. Cria o gerador de fluxo (nenhuma amostra é realmente criada aqui ainda)
            var samplesStream = StreamAllSamplesAsJson(text, contextWindowSize, tokenToIndex, padToken);

            // 2. Define a função de callback para o progresso
            Action<int> progressCallback = count =>
            {
                Console.Write($"\rArmazenando amostras... {count} salvas.");
            };

            // 3. Passa o fluxo e o callback para o método de armazenamento otimizado
            // O armazenamento irá puxar as amostras do fluxo uma a uma, mantendo a memória baixa.
            var sampleOffsets = memoryStorage.StreamAndStoreSamples(samplesStream, progressCallback);

            Console.WriteLine($"Bloco processado. Total de {sampleOffsets.Count} amostras armazenadas.");
            return sampleOffsets;
        }
        
        private IEnumerable<string> StreamAllSamplesAsJson(string text, int contextWindowSize, Dictionary<string, int> tokenToIndex, string padToken)
        {
            var paddedTokens = TokenizeAndPad(text, contextWindowSize, padToken);
            int totalWindows = paddedTokens.Count - contextWindowSize;

            for (int i = 0; i < totalWindows; i++)
            {
                var sample = CreateSample(paddedTokens, i, contextWindowSize, tokenToIndex, padToken);
                if (sample.HasValue)
                {
                    var (input, target) = sample.Value;
                    var sampleData = new TensorPairData
                    {
                        Input = new TensorData { data = input.GetData(), shape = input.GetShape() },
                        Target = new TensorData { data = target.GetData(), shape = target.GetShape() }
                    };
                    string serializedSample = JsonSerializer.Serialize(sampleData);

                    if (System.Text.Encoding.UTF8.GetByteCount(serializedSample) <= TreeNode.MaxDataSize)
                    {
                        // 'yield return' produz um item para o consumidor e pausa a execução
                        // até que o próximo item seja solicitado. Isso evita criar uma lista.
                        yield return serializedSample;
                    }
                }
            }
        }

        // Recupera uma ÚNICA amostra da memória virtual
        public (Tensor input, Tensor target) GetSampleFromStorage(long offset, BinaryTreeFileStorage memoryStorage)
        {
            string serializedSample = memoryStorage.GetData(offset);
            var sampleData = JsonSerializer.Deserialize<TensorPairData>(serializedSample);

            if (sampleData == null)
            {
                throw new InvalidDataException($"Não foi possível desserializar a amostra no offset {offset}.");
            }

            return (new Tensor(sampleData.Input.data, sampleData.Input.shape),
                    new Tensor(sampleData.Target.data, sampleData.Target.shape));
        }
        
        // Método auxiliar PRIVADO que faz o trabalho pesado de forma paralela.
        private List<(Tensor input, Tensor target)> GenerateAllSamples(string text, int contextWindowSize, Dictionary<string, int> tokenToIndex, string padToken)
        {
            var paddedTokens = TokenizeAndPad(text, contextWindowSize, padToken);
            var concurrentDataset = new ConcurrentBag<(Tensor input, Tensor target)>();
            int totalWindows = paddedTokens.Count - contextWindowSize;

            Parallel.For(0, totalWindows, i =>
            {
                var sample = CreateSample(paddedTokens, i, contextWindowSize, tokenToIndex, padToken);
                if (sample.HasValue)
                {
                    concurrentDataset.Add(sample.Value);
                }
            });

            return concurrentDataset.ToList();
        }

        private List<string> TokenizeAndPad(string text, int contextWindowSize, string padToken)
        {
            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            var tokens = Regex.Matches(text.ToLower(), pattern).Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToArray();
            
            var paddedTokens = new List<string>(tokens.Length + contextWindowSize);
            for (int k = 0; k < contextWindowSize; k++) { paddedTokens.Add(padToken); }
            paddedTokens.AddRange(tokens);
            return paddedTokens;
        }

        private (Tensor input, Tensor target)? CreateSample(List<string> paddedTokens, int index, int contextWindowSize, Dictionary<string, int> tokenToIndex, string padToken)
        {
            string nextToken = paddedTokens[index + contextWindowSize];
            if (!tokenToIndex.ContainsKey(nextToken)) return null;

            int inputSize = tokenToIndex.Count * contextWindowSize;
            double[] inputData = _arrayPool.Rent(inputSize);
            
            try
            {
                Array.Clear(inputData, 0, inputSize);
                for (int k = 0; k < contextWindowSize; k++)
                {
                    string token = paddedTokens[index + k];
                    if (!tokenToIndex.TryGetValue(token, out int tokenVocabIndex))
                    {
                        return null;
                    }
                    inputData[k * tokenToIndex.Count + tokenVocabIndex] = 1.0;
                }
                
                int targetSize = tokenToIndex.Count;
                double[] targetData = _arrayPool.Rent(targetSize);
                try
                {
                    Array.Clear(targetData, 0, targetSize);
                    targetData[tokenToIndex[nextToken]] = 1.0;
                    
                    var finalInputData = new double[inputSize];
                    Array.Copy(inputData, finalInputData, inputSize);
                    
                    var finalTargetData = new double[targetSize];
                    Array.Copy(targetData, finalTargetData, targetSize);

                    var inputTensor = new Tensor(finalInputData, new int[] { inputSize });
                    var targetTensor = new Tensor(finalTargetData, new int[] { targetSize });

                    return (inputTensor, targetTensor);
                }
                finally
                {
                    _arrayPool.Return(targetData);
                }
            }
            finally
            {
                _arrayPool.Return(inputData);
            }
        }
        
        public List<(Tensor input, Tensor target)> PrepareSingleBatch(string text, int contextWindowSize, Dictionary<string, int> tokenToIndex, string padToken)
        {
            return GenerateAllSamples(text, contextWindowSize, tokenToIndex, padToken);
        }
        
        public List<List<(Tensor input, Tensor target)>> PrepareBatchesFromText(
            string text, 
            int contextWindowSize, 
            Dictionary<string, int> tokenToIndex, 
            string padToken, 
            int batchSize)
        {
            var allSamples = GenerateAllSamples(text, contextWindowSize, tokenToIndex, padToken);
            var batches = new List<List<(Tensor input, Tensor target)>>();

            for (int i = 0; i < allSamples.Count; i += batchSize)
            {
                var batch = allSamples.Skip(i).Take(batchSize).ToList();
                if (batch.Count > 0)
                {
                    batches.Add(batch);
                }
            }
            return batches;
        }
    }
}
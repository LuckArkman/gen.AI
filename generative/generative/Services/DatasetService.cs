using System;
using System.Buffers;
using System.Collections.Concurrent; // Necessário para ConcurrentBag
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks; // Necessário para Parallel.ForEach
using Core;

namespace Services
{
    public class DatasetService
    {
        private static readonly ArrayPool<double> _arrayPool = ArrayPool<double>.Shared;

        public List<(Tensor input, Tensor target)> PrepareDataset(string text, int contextWindowSize, Dictionary<string, int> tokenToIndex, string padToken)
        {
            // 1. Tokenização (rápida, pode continuar no thread principal)
            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            var tokens = Regex.Matches(text.ToLower(), pattern).Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToArray();
            
            var paddedTokens = new List<string>(tokens.Length + contextWindowSize);
            for (int k = 0; k < contextWindowSize; k++) { paddedTokens.Add(padToken); }
            paddedTokens.AddRange(tokens);

            // 2. Paralelização do Processamento de Janelas
            // Usamos um ConcurrentBag para coletar os resultados de múltiplos threads de forma segura.
            var concurrentDataset = new ConcurrentBag<(Tensor input, Tensor target)>();
            
            int totalWindows = paddedTokens.Count - contextWindowSize;

            // Enumerable.Range cria uma sequência de números de 0 a totalWindows-1.
            // .AsParallel() ativa o PLINQ, distribuindo o trabalho entre os núcleos da CPU.
            Parallel.For(0, totalWindows, i =>
            {
                // Cada iteração do loop agora pode rodar em um thread diferente.
                
                // Pega a janela e o próximo token
                var currentWindowTokens = new ArraySegment<string>(paddedTokens.ToArray(), i, contextWindowSize);
                string nextToken = paddedTokens[i + contextWindowSize];

                // Valida se o próximo token existe no vocabulário
                if (!tokenToIndex.ContainsKey(nextToken))
                {
                    return; // 'return' em um Parallel.For é como 'continue' em um for normal
                }

                int inputSize = tokenToIndex.Count * contextWindowSize;
                double[] inputData = _arrayPool.Rent(inputSize);
                try
                {
                    Array.Clear(inputData, 0, inputSize);
                    bool allTokensValid = true;

                    // Valida e preenche o one-hot encoding para a janela
                    for (int k = 0; k < contextWindowSize; k++)
                    {
                        string token = currentWindowTokens[k];
                        if (!tokenToIndex.TryGetValue(token, out int tokenVocabIndex))
                        {
                            allTokensValid = false;
                            break;
                        }
                        inputData[k * tokenToIndex.Count + tokenVocabIndex] = 1.0;
                    }

                    if (!allTokensValid)
                    {
                        return; // Pula esta janela se um token for inválido
                    }

                    // Cria o vetor de target (one-hot)
                    int targetSize = tokenToIndex.Count;
                    double[] targetData = _arrayPool.Rent(targetSize);
                    try
                    {
                        Array.Clear(targetData, 0, targetSize);
                        targetData[tokenToIndex[nextToken]] = 1.0;

                        // Copia os dados dos arrays alugados para arrays finais
                        var finalInputData = new double[inputSize];
                        Array.Copy(inputData, finalInputData, inputSize);

                        var finalTargetData = new double[targetSize];
                        Array.Copy(targetData, finalTargetData, targetSize);

                        // Cria os Tensors e adiciona ao ConcurrentBag
                        var inputTensor = new Tensor(finalInputData, new int[] { inputSize });
                        var targetTensor = new Tensor(finalTargetData, new int[] { targetSize });
                        concurrentDataset.Add((inputTensor, targetTensor));
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
            });

            // 3. Converte o ConcurrentBag de volta para uma List para o retorno.
            return concurrentDataset.ToList();
        }
    }
}
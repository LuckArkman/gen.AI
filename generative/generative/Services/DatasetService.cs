using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Core;

namespace Services
{
    public class DatasetService
    {
        public List<(Tensor input, Tensor target)> PrepareDataset(string text, int contextWindowSize, Dictionary<string, int> tokenToIndex, string padToken)
        {
            var dataset = new List<(Tensor input, Tensor target)>();
            var specialChars = new[] { '.', ',', '!', '?', ':', ';', '"', '\'', '-', '(', ')' };
            var specialCharPattern = string.Join("|", specialChars.Select(c => Regex.Escape(c.ToString())));
            var pattern = $@"(\p{{L}}+|\p{{N}}+|{specialCharPattern})";
            var tokens = Regex.Matches(text.ToLower(), pattern).Select(m => m.Value).Where(t => !string.IsNullOrEmpty(t)).ToArray();
            var paddedTokens = new List<string>();
            for (int k = 0; k < contextWindowSize; k++) { paddedTokens.Add(padToken); }
            paddedTokens.AddRange(tokens);

            for (int i = 0; i < paddedTokens.Count - contextWindowSize; i++)
            {
                string[] currentWindowTokens = paddedTokens.Skip(i).Take(contextWindowSize).ToArray();
                string nextToken = paddedTokens[i + contextWindowSize];

                if (!tokenToIndex.ContainsKey(nextToken) || !currentWindowTokens.All(t => tokenToIndex.ContainsKey(t)))
                {
                    continue;
                }

                double[] inputData = new double[tokenToIndex.Count * contextWindowSize];
                for (int k = 0; k < contextWindowSize; k++)
                {
                    int tokenVocabIndex = tokenToIndex[currentWindowTokens[k]];
                    int offset = k * tokenToIndex.Count;
                    inputData[offset + tokenVocabIndex] = 1.0;
                }
                var inputTensor = new Tensor(inputData, new int[] { tokenToIndex.Count * contextWindowSize });

                double[] targetData = new double[tokenToIndex.Count];
                targetData[tokenToIndex[nextToken]] = 1.0;
                var targetTensor = new Tensor(targetData, new int[] { tokenToIndex.Count });
                
                dataset.Add((inputTensor, targetTensor));
            }
            return dataset;
        }
    }
}
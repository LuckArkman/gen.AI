// Services/TextProcessorService.cs
using System.Linq;

namespace Services
{
    public class TextProcessorService
    {
        public string Summarize(string text, int maxWords = 50)
        {
            if (string.IsNullOrWhiteSpace(text)) return string.Empty;

            var words = text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length <= maxWords) return text;

            return string.Join(" ", words.Take(maxWords)) + "...";
        }
        public string GenerateContextHash(string text)
        {
            using (var sha256 = System.Security.Cryptography.SHA256.Create())
            {
                byte[] bytes = System.Text.Encoding.UTF8.GetBytes(text);
                byte[] hashBytes = sha256.ComputeHash(bytes);
                return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
            }
        }
        public string ExtractMainTopic(string text, int maxTopicLength = 200)
        {
            if (string.IsNullOrWhiteSpace(text)) return string.Empty;
            var words = text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            return Summarize(string.Join(" ", words.Take(20)), maxTopicLength);
        }
    }
}
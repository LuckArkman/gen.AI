using AlphaOne.Models;
using MongoDB.Driver;

namespace AlphaOne.Services;

public class ConversationService
    {
        private readonly IMongoCollection<Conversation> _conversations;

        public ConversationService(IConfiguration configuration)
        {
            var client = new MongoClient(configuration.GetConnectionString("MongoDbConnection"));
            var database = client.GetDatabase(configuration["ConnectionStrings:MongoDbName"]);
            _conversations = database.GetCollection<Conversation>("Conversations");
        }

        public async Task<List<Conversation>> GetConversationsByUserIdAsync(string userId)
        {
            return await _conversations.Find(c => c.UserId == userId)
                                       .SortByDescending(c => c.LastUpdated)
                                       .ToListAsync();
        }

        public async Task<Conversation?> GetConversationByIdAsync(string conversationId, string userId)
        {
            return await _conversations.Find(c => c.Id == conversationId && c.UserId == userId).FirstOrDefaultAsync();
        }

        public async Task CreateConversationAsync(Conversation conversation)
        {
            await _conversations.InsertOneAsync(conversation);
        }

        public async Task AddMessageToConversationAsync(string userId, string? conversationId, string userMessage, string aiResponse)
        {
            Conversation? conversation;

            if (string.IsNullOrEmpty(conversationId))
            {
                conversation = new Conversation
                {
                    UserId = userId,
                    Title = userMessage.Length > 50 ? userMessage.Substring(0, 50) + "..." : userMessage
                };
                conversation.Messages.Add(new Message { Sender = "user", Text = userMessage });
                conversation.Messages.Add(new Message { Sender = "ai", Text = aiResponse });
                await CreateConversationAsync(conversation);
                Console.WriteLine($"Nova conversa criada para {userId} com título '{conversation.Title}'. ID: {conversation.Id}");
            }
            else
            {
                conversation = await GetConversationByIdAsync(conversationId, userId);
                if (conversation != null)
                {
                    conversation.Messages.Add(new Message { Sender = "user", Text = userMessage });
                    conversation.Messages.Add(new Message { Sender = "ai", Text = aiResponse });
                    conversation.LastUpdated = DateTime.UtcNow;
                    await _conversations.ReplaceOneAsync(c => c.Id == conversation.Id, conversation);
                    Console.WriteLine($"Mensagens adicionadas à conversa {conversation.Id} para {userId}.");
                }
                else
                {
                    Console.WriteLine($"Aviso: ConversationId {conversationId} não encontrado para o usuário {userId}. Criando nova conversa.");
                    conversation = new Conversation
                    {
                        UserId = userId,
                        Title = userMessage.Length > 50 ? userMessage.Substring(0, 50) + "..." : userMessage
                    };
                    conversation.Messages.Add(new Message { Sender = "user", Text = userMessage });
                    conversation.Messages.Add(new Message { Sender = "ai", Text = aiResponse });
                    await CreateConversationAsync(conversation);
                }
            }
        }
    }
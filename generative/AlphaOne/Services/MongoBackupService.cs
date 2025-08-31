using AlphaOne.Data;
using MongoDB.Driver;

namespace AlphaOne.Services;

public class MongoBackupService
{
    private readonly IMongoCollection<BackupUser> _users;

    public MongoBackupService(IConfiguration configuration)
    {
        var client = new MongoClient(configuration.GetConnectionString("MongoDbConnection"));
        var database = client.GetDatabase(configuration["ConnectionStrings:MongoDbName"]);
        _users = database.GetCollection<BackupUser>("Users");
    }

    public async Task BackupUserAsync(ApplicationUser user)
    {
        var backupUser = new BackupUser
        {
            Id = user.Id,
            UserName = user.UserName,
            Email = user.Email,
            PhoneNumber = user.PhoneNumber,
            PasswordHash = user.PasswordHash,
            PersonalName = user.PersonalName,
            Nickname = user.Nickname,
            BackupDate = DateTime.UtcNow
        };

        await _users.ReplaceOneAsync(u => u.Id == backupUser.Id, backupUser, new ReplaceOptions { IsUpsert = true });
        Console.WriteLine($"Usuário {user.UserName} backup para MongoDB.");
    }

    public class BackupUser
    {
        public string Id { get; set; } = string.Empty;
        public string? UserName { get; set; }
        public string? Email { get; set; }
        public string? PhoneNumber { get; set; }
        public string? PasswordHash { get; set; }
        public string? PersonalName { get; set; }
        public string? Nickname { get; set; }
        public DateTime BackupDate { get; set; }
    }
}
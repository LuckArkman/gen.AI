using Microsoft.AspNetCore.Identity;

namespace AlphaOne.Data;

public class ApplicationUser : IdentityUser
{
    public string? PersonalName { get; set; }
    public string? Nickname { get; set; }
}
// GenerativeAIWebApp/Controllers/DevController.cs (NOVO ARQUIVO)
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Hosting; // Para IWebHostEnvironment
using AlphaOne.Data;
using System.Threading.Tasks;

namespace AlphaOne.Controllers
{
    [ApiController] // Este é um controller de API, não de View
    [Route("api/[controller]")]
    public class DevController : ControllerBase
    {
        private readonly SignInManager<ApplicationUser> _signInManager;
        private readonly UserManager<ApplicationUser> _userManager;
        private readonly IWebHostEnvironment _env;

        public DevController(SignInManager<ApplicationUser> signInManager, UserManager<ApplicationUser> userManager, IWebHostEnvironment env)
        {
            _signInManager = signInManager;
            _userManager = userManager;
            _env = env;
        }

        [HttpGet("autologin")]
        public async Task<IActionResult> AutoLogin(string username, string password)
        {
            if (!_env.IsDevelopment())
            {
                return Forbid("This endpoint is only available in Development environment.");
            }

            var user = await _userManager.FindByNameAsync(username);
            if (user == null)
            {
                user = await _userManager.FindByEmailAsync(username); // Tentar por email
            }

            if (user == null)
            {
                // Opcional: Criar usuário de dev se não existir
                user = new ApplicationUser { UserName = username, Email = $"{username}@example.com", PersonalName = "Dev", Nickname = "DevUser" };
                var createResult = await _userManager.CreateAsync(user, password);
                if (!createResult.Succeeded)
                {
                    return BadRequest($"Failed to create dev user: {string.Join(", ", createResult.Errors.Select(e => e.Description))}");
                }
            }

            var result = await _signInManager.PasswordSignInAsync(user, password, isPersistent: true, lockoutOnFailure: false);

            if (result.Succeeded)
            {
                return Ok("Logged in successfully as dev user.");
            }
            else if (result.IsLockedOut)
            {
                return BadRequest("Dev user account locked out.");
            }
            else
            {
                return BadRequest("Invalid credentials for dev user or other login issue.");
            }
        }
    }
}
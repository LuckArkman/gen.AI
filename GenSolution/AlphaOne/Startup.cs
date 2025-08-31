// GenerativeAIWebApp/Startup.cs
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using AlphaOne.Data;
using AlphaOne.Services;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using AlphaOne.Middlewares; // <-- Adicionar este using

namespace AlphaOne
{
    public class Startup
    {
        // ... (restante do Startup.cs)

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            // ... (DeveloperExceptionPage, UseExceptionHandler, etc.)

            app.UseHttpsRedirection();
            app.UseStaticFiles();

            app.UseRouting();

            // Adicione o middleware de bypass de autenticação AQUI
            // Ele deve vir ANTES de UseAuthentication e UseAuthorization
            // para que o HttpContext.User seja modificado antes que a autenticação seja checada.
            app.UseMiddleware<DevBypassAuthMiddleware>();

            app.UseAuthentication(); // Sempre aqui
            app.UseAuthorization();  // Sempre aqui

            app.UseEndpoints(endpoints =>
            {
                // Rota padrão para Login (em produção) ou Home/Index (em dev via bypass)
                // Não há mais necessidade de ifs aqui, o middleware se encarrega.
                endpoints.MapControllerRoute(
                    name: "default",
                    pattern: "{controller=Account}/{action=Login}/{id?}"); // Redireciona para login se não autenticado

                // Rota para o Home/Index (seja via login ou via bypass)
                endpoints.MapControllerRoute(
                    name: "chat",
                    pattern: "Home/{action=Index}/{id?}",
                    defaults: new { controller = "Home", action = "Index" }
                );
            });
        }
    }
}
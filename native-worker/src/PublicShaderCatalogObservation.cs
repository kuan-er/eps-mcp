using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace SprayBot.EpsNativeStateProbe;

// Candidate only: not compiled, deployed, or executed. Add to the existing
// normal public-state snapshot on its normal UI dispatcher, NEVER to the
// FirstChanceException handler. Does not construct a provider or runtime.
internal static class PublicShaderCatalogObservation
{
    private const string CatalogType = "Olpp.Infrastructure.Eyeshot.Painting.PaintShaderCatalogService";

    internal static Dictionary<string, object?> Capture(IServiceProvider services)
    {
        var r = new Dictionary<string, object?> {
            ["schema"] = "eps-public-shader-catalog-observation/1",
            ["utc"] = DateTime.UtcNow,
            ["service_type"] = CatalogType,
            ["source"] = "existing public App.ServiceProvider.GetService(Type), public ShaderDirectory getter",
            ["shader_directory"] = null,
            ["running_file_shader_provider_identity_verified"] = false,
            ["actual_compiler_source_verified"] = false,
            ["native_execution_or_deposition_verified"] = false,
            ["limit"] = "A public catalog observation is not direct observation of the FileShaderSourceProvider instance used by the failed compiler."
        };
        try
        {
            var candidates = AppDomain.CurrentDomain.GetAssemblies()
                .Select(a => a.GetType(CatalogType, false, false))
                .Where(t => t != null && t.IsVisible).ToArray();
            r["public_loaded_type_matches"] = candidates.Length;
            if (candidates.Length != 1) { r["state"] = "missing_or_ambiguous_public_catalog_type"; return r; }
            Type type = candidates[0]!;
            var property = type.GetProperty("ShaderDirectory", BindingFlags.Public | BindingFlags.Instance);
            if (property?.GetMethod?.IsPublic != true || property.GetIndexParameters().Length != 0
                || property.PropertyType != typeof(string))
            { r["state"] = "expected_public_string_getter_unavailable"; return r; }
            // Normal DI resolution can invoke a registered factory. Never use
            // Activator or an unregistered constructor, and retain that limit.
            r["registered_factory_resolution_possible"] = true;
            object? first = services.GetService(type);
            object? second = services.GetService(type);
            r["service_resolved"] = first != null;
            r["same_instance_on_repeated_resolution"] = first != null && ReferenceEquals(first, second);
            if (first == null) { r["state"] = "catalog_not_resolved_from_existing_provider"; return r; }
            if (!ReferenceEquals(first, second))
            { r["state"] = "unstable_catalog_resolution_no_value_claim"; return r; }
            string? value = property.GetValue(first) as string;
            r["shader_directory"] = value == null ? null : value.Substring(0, Math.Min(value.Length, 2048));
            r["directory_truncated"] = value != null && value.Length > 2048;
            r["catalog_assembly"] = type.Assembly.GetName().Name;
            r["state"] = "stable_public_catalog_observed";
        }
        catch (Exception ex)
        {
            r["state"] = "observation_failed";
            r["error_type"] = ex.GetType().FullName;
            string message = ex.Message ?? "";
            r["error"] = message.Substring(0, Math.Min(message.Length, 256));
        }
        return r;
    }
}

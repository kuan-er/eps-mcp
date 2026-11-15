using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

namespace SprayBot.EpsNativeStateProbe;

// Optional observation helper, not yet compiled or deployed. Call only from an
// explicit bounded recording's existing guarded painting-exception observer.
// Uses public Win32/WGL/OpenGL queries from an ALREADY loaded opengl32 module.
// Never loads a graphics provider, creates/makes-current a context, compiles a
// shader, changes state, reads glGetError, or accesses a private EPS member.
internal static class NativeGlObservation
{
    private const uint GL_VENDOR = 0x1F00, GL_RENDERER = 0x1F01, GL_VERSION = 0x1F02;
    private const uint GL_SHADING_LANGUAGE_VERSION = 0x8B8C;
    private const uint GL_CONTEXT_PROFILE_MASK = 0x9126;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern IntPtr GetModuleHandleW(string moduleName);

    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, ExactSpelling = true)]
    private static extern IntPtr GetProcAddress(IntPtr module, string name);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern uint GetModuleFileNameW(IntPtr module, StringBuilder path, int capacity);

    [DllImport("kernel32.dll", ExactSpelling = true)]
    private static extern uint GetCurrentThreadId();

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr CurrentContext();

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate IntPtr GetString(uint name);

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate void GetInteger(uint name, out int value);

    private static string? BoundedString(IntPtr p)
    {
        if (p == IntPtr.Zero) return null;
        int length = 0;
        while (length < 512 && Marshal.ReadByte(p, length) != 0) length++;
        return Marshal.PtrToStringAnsi(p, length);
    }

    private static T? Export<T>(IntPtr module, string name) where T : Delegate
    {
        var address = GetProcAddress(module, name);
        return address == IntPtr.Zero ? null : Marshal.GetDelegateForFunctionPointer<T>(address);
    }

    internal static Dictionary<string, object?> CaptureOnThisThread()
    {
        var r = new Dictionary<string, object?> {
            ["schema"] = "eps-passive-gl-thread-observation/1",
            ["utc"] = DateTime.UtcNow,
            ["managed_thread_id"] = Environment.CurrentManagedThreadId,
            ["scope"] = "public queries on the observer callback thread using already loaded opengl32 exports",
            ["context_or_shader_state_changed"] = false,
            ["eps_viewport_context_identity_verified"] = false,
            ["compiler_error_or_compatibility_support_verified"] = false,
            ["gl_version"] = null, ["glsl_version"] = null,
            ["gl_vendor"] = null, ["gl_renderer"] = null,
            ["context_profile_mask"] = null, ["context_profile"] = null
        };
        if (!OperatingSystem.IsWindows()) { r["state"] = "not_windows"; return r; }
        try
        {
            r["native_thread_id"] = GetCurrentThreadId();
            // GetModuleHandle does not load a DLL. Never substitute a separately
            // loaded system or application-local provider when none is present.
            var module = GetModuleHandleW("opengl32.dll");
            r["opengl32_already_loaded"] = module != IntPtr.Zero;
            if (module == IntPtr.Zero) { r["state"] = "no_loaded_opengl32"; return r; }
            var path = new StringBuilder(1024);
            uint pathLength = GetModuleFileNameW(module, path, path.Capacity);
            r["opengl32_module_path"] = pathLength > 0 && pathLength < path.Capacity ? path.ToString() : null;
            var current = Export<CurrentContext>(module, "wglGetCurrentContext");
            var text = Export<GetString>(module, "glGetString");
            if (current == null || text == null) { r["state"] = "public_exports_unavailable"; return r; }
            var before = current();
            r["current_wgl_context_on_this_thread"] = before != IntPtr.Zero;
            if (before == IntPtr.Zero)
            {
                r["state"] = "no_current_wgl_context_on_observer_thread";
                r["limit"] = "No current WGL context is observed; this alone does not prove the EPS compile failure cause or exclude a different context API.";
                return r;
            }
            string? version = BoundedString(text(GL_VERSION));
            r["gl_version"] = version;
            r["gl_vendor"] = BoundedString(text(GL_VENDOR));
            r["gl_renderer"] = BoundedString(text(GL_RENDERER));
            bool desktop = version != null && !version.StartsWith("OpenGL ES", StringComparison.Ordinal);
            var fields = (version ?? "").Split(' ', 2)[0].Split('.');
            int major = 0, minor = 0;
            bool parsed = desktop && fields.Length >= 2
                && int.TryParse(fields[0], NumberStyles.None, CultureInfo.InvariantCulture, out major)
                && int.TryParse(fields[1], NumberStyles.None, CultureInfo.InvariantCulture, out minor);
            if (parsed && major >= 2)
                r["glsl_version"] = BoundedString(text(GL_SHADING_LANGUAGE_VERSION));
            // Profile mask is a desktop GL3.2+ query. Do not issue unsupported
            // enum queries or consume the host's error queue with glGetError.
            if (parsed && (major > 3 || major == 3 && minor >= 2))
            {
                var integer = Export<GetInteger>(module, "glGetIntegerv");
                if (integer != null)
                {
                    integer(GL_CONTEXT_PROFILE_MASK, out int mask);
                    r["context_profile_mask"] = mask;
                    r["context_profile"] = mask == 1 ? "core" : mask == 2 ? "compatibility" : "unknown";
                }
            }
            else r["profile_query_skipped"] = "No parseable desktop GL3.2+ version; no profile enum query issued.";
            r["same_current_wgl_context_before_after"] = current() == before;
            r["state"] = version == null ? "current_wgl_context_but_no_version_string" : "observed";
            r["limit"] = "This is a passive observation, not an EPS shader compiler log or proof of source resolution, shader support, native painting, or deposition.";
        }
        catch (Exception ex)
        {
            r["state"] = "observation_failed";
            r["error_type"] = ex.GetType().FullName;
            string message = ex.Message ?? "";
            r["error"] = message.Length > 256 ? message.Substring(0, 256) : message;
        }
        return r;
    }
}

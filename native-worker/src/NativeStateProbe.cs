using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows;
using System.Windows.Automation.Peers;
using System.Windows.Automation.Provider;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using System.Windows.Input;
using Olpp.Module;

namespace SprayBot.EpsNativeStateProbe;

// EPS IModule extension for the isolated offline benchmark. Use the normal
// recording service and the command behind the loaded simulation toolbar.
public sealed class NativeStateProbe : IModule
{
    private readonly IHostingApplication host;
    private readonly string instance = Guid.NewGuid().ToString("N");
    private readonly HashSet<string> seen = new();
    private Config? config;
    private DispatcherTimer? timer;
    private Window? window;
    private TextBox? display;
    private string last = "Waiting for the configured isolated project";
    private bool readyWritten;
    private long requestStamp;
    private StreamWriter? trace;
    private string? observationDir;
    private Stopwatch? elapsed;
    private long nextFrame;
    private int durationMs;
    private int intervalMs;
    private int frames;
    private bool stopping;
    private bool recordingArmedByBridge;
    private bool simulationDispatchedByBridge;
    private readonly ConcurrentQueue<object> caughtPaintingExceptions = new();
    private volatile bool observePaintingExceptions;
    private int exceptionCandidates;
    private int exceptionObservations;
    private int exceptionObservationLimitDrops;
    private int exceptionEventsWritten;
    private int exceptionEventsDropped;
    [ThreadStatic] private static bool insideExceptionObserver;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = false };

    public NativeStateProbe(IHostingApplication application)
    {
        host = application;
        try
        {
            var moduleDir = Path.GetDirectoryName(typeof(NativeStateProbe).Assembly.Location)!;
            config = JsonSerializer.Deserialize<Config>(File.ReadAllText(Path.Combine(moduleDir, "probe-config.json")))
                ?? throw new InvalidDataException("Missing probe configuration");
            config.AllowedWorkRoot = Canonical(config.AllowedWorkRoot);
            config.AllowedProjectFolder = Canonical(config.AllowedProjectFolder);
            config.OutputRoot = Canonical(config.OutputRoot);
            if (!Under(config.AllowedProjectFolder, config.AllowedWorkRoot) || !Under(config.OutputRoot, config.AllowedWorkRoot))
                throw new InvalidDataException("Project and output must stay inside the configured task work root");
            Directory.CreateDirectory(config.OutputRoot);
            WriteNew(Path.Combine(config.OutputRoot, "loaded-" + Environment.ProcessId + "-" + instance + ".json"), new {
                kind = "normal_module_constructor_observed", utc = DateTime.UtcNow, eps_pid = Environment.ProcessId,
                module_version = "2.2-recording-adapter", module_instance = instance, dll_sha256 = Hash(typeof(NativeStateProbe).Assembly.Location),
                source = "normal IModule constructor", initialized_simulator = false, called_controller = false
            });
            AppDomain.CurrentDomain.FirstChanceException += ObserveCaughtPaintingException;
            Schedule();
        }
        catch (Exception ex) { last = Describe(ex); config = null; }
    }

    public string Name => "SprayBot public native state probe";
    public bool IsDisposed { get; private set; }
    public void Init() => Schedule();
    public object GetWindow() => Ui(() => { EnsureWindow(); return (object)window!; });
    public void ShowWindow() => Ui(() => { EnsureWindow(); window!.Show(); return 0; });
    public void HideWindow() => Ui(() => { window?.Hide(); return 0; });
    public bool IsWindowVisible() => Ui(() => window?.IsVisible ?? false);
    public void AnimationStarted() => Event("module_animation_started");
    public void AnimationEnded() => Event("module_animation_ended");
    public void Close() => Ui(() => { Stop("module_closed"); AppDomain.CurrentDomain.FirstChanceException -= ObserveCaughtPaintingException; timer?.Stop(); timer = null; IsDisposed = true; window?.Close(); return 0; });

    // Public CLR event, active only during a guarded explicit recording. No service
    // resolution, UI/host reads or file IO on the throwing thread; bounded queue.
    private void ObserveCaughtPaintingException(object? sender, FirstChanceExceptionEventArgs args)
    {
        if (!observePaintingExceptions || insideExceptionObserver) return;
        insideExceptionObserver = true;
        try
        {
            if (Interlocked.Increment(ref exceptionObservations) > 512)
            { Interlocked.Increment(ref exceptionObservationLimitDrops); return; }
            string exceptionStack = args.Exception.StackTrace ?? "";
            string callerStack = new StackTrace(1, false).ToString();
            const string paintingPrefix = "Olpp.Infrastructure.Eyeshot.Painting.";
            string exceptionMessage = args.Exception.Message ?? "";
            bool shaderRelated = exceptionMessage.Contains("shader", StringComparison.OrdinalIgnoreCase)
                || exceptionMessage.Contains("GLSL", StringComparison.OrdinalIgnoreCase)
                || exceptionMessage.Contains("OpenGL", StringComparison.OrdinalIgnoreCase);
            if (!shaderRelated && !exceptionStack.Contains(paintingPrefix, StringComparison.Ordinal)
                && !callerStack.Contains(paintingPrefix, StringComparison.Ordinal)) return;
            if (Interlocked.Increment(ref exceptionCandidates) > 32)
            { Interlocked.Increment(ref exceptionEventsDropped); return; }
            string message = args.Exception.Message ?? "";
            caughtPaintingExceptions.Enqueue(new {
                kind = "caught_native_painting_exception", utc = DateTime.UtcNow,
                monotonic_ticks = Stopwatch.GetTimestamp(), eps_pid = Environment.ProcessId,
                module_instance = instance, managed_thread_id = Environment.CurrentManagedThreadId,
                exception_type = args.Exception.GetType().FullName,
                message = message.Length > 2048 ? message.Substring(0, 2048) : message,
                exception_stack = exceptionStack.Length > 8192 ? exceptionStack.Substring(0, 8192) : exceptionStack,
                caller_stack = callerStack.Length > 8192 ? callerStack.Substring(0, 8192) : callerStack,
                gl_thread_observation = NativeGlObservation.CaptureOnThisThread(),
                message_truncated = message.Length > 2048,
                exception_stack_truncated = exceptionStack.Length > 8192,
                caller_stack_truncated = callerStack.Length > 8192,
                source = "AppDomain.FirstChanceException public event; exception or caller stack filtered to native painting",
                implication = "An exception was thrown; causation and ultimate handling remain to be established"
            });
        }
        catch { /* Diagnostics must not alter host exception handling. */ }
        finally { insideExceptionObserver = false; }
    }
    private void DrainCaughtPaintingExceptions()
    {
        while (caughtPaintingExceptions.TryDequeue(out var entry))
        {
            if (trace == null) { Interlocked.Increment(ref exceptionEventsDropped); continue; }
            string line = JsonSerializer.Serialize(entry, Json);
            if (trace.BaseStream.Position + Encoding.UTF8.GetByteCount(line) + 2 > 1024 * 1024)
            { Interlocked.Increment(ref exceptionEventsDropped); continue; }
            trace.WriteLine(line); trace.Flush(); exceptionEventsWritten++;
        }
    }

    private static T Ui<T>(Func<T> action)
    {
        var dispatcher = Application.Current?.Dispatcher ?? throw new InvalidOperationException("EPS dispatcher unavailable");
        return dispatcher.CheckAccess() ? action() : dispatcher.Invoke(action);
    }
    private void Schedule()
    {
        var dispatcher = Application.Current?.Dispatcher;
        if (config == null || dispatcher == null || dispatcher.HasShutdownStarted || IsDisposed) return;
        dispatcher.BeginInvoke(new Action(() => {
            if (timer != null || IsDisposed) return;
            timer = new DispatcherTimer(DispatcherPriority.Background) { Interval = TimeSpan.FromMilliseconds(50) };
            timer.Tick += (_, _) => Guard(Tick);
            timer.Start();
        }));
    }
    private void EnsureWindow()
    {
        if (window != null) return;
        display = new TextBox { IsReadOnly = true, AcceptsReturn = true, TextWrapping = TextWrapping.NoWrap,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, Text = last };
        window = new Window { Title = Name, Width = 950, Height = 600, Content = display };
        window.Closing += (_, e) => { if (!IsDisposed) { e.Cancel = true; window.Hide(); } };
    }
    private void Guard(Action action)
    {
        try { action(); }
        catch (Exception ex)
        {
            last = Describe(ex);
            Stop("observer_error");
            try { if (config != null) WriteNew(Path.Combine(config.OutputRoot, "error-" + Guid.NewGuid().ToString("N") + ".json"), new { utc = DateTime.UtcNow, error = last }); }
            catch { /* Keep host usable even if the task output disk fails. */ }
            if (display != null) display.Text = last;
        }
    }
    private bool ProjectMatches()
    {
        try { return config != null && Canonical(host.ProjectFolderPath).Equals(config.AllowedProjectFolder, StringComparison.OrdinalIgnoreCase); }
        catch { return false; }
    }
    private void Tick()
    {
        if (config == null || IsDisposed) return;
        if (!ProjectMatches()) { if (trace != null) Stop("active_project_changed"); return; }
        if (trace != null) DrainCaughtPaintingExceptions();
        if (!readyWritten)
        {
            var snapshot = Capture(true);
            WriteNew(Path.Combine(config.OutputRoot, "ready-" + Environment.ProcessId + "-" + instance + ".json"), snapshot);
            readyWritten = true;
        }
        if (trace != null && Stopwatch.GetTimestamp() >= nextFrame)
        {
            Append("sample", false);
            nextFrame = Stopwatch.GetTimestamp() + (long)(Stopwatch.Frequency * intervalMs / 1000.0);
            if (elapsed!.ElapsedMilliseconds >= durationMs) Stop("duration_reached");
        }
        var requestPath = Path.Combine(config.OutputRoot, "request.json");
        if (!File.Exists(requestPath)) return;
        var info = new FileInfo(requestPath);
        if (info.LastWriteTimeUtc.Ticks == requestStamp) return;
        requestStamp = info.LastWriteTimeUtc.Ticks;
        if (info.Length > 4096) throw new InvalidDataException("Probe request exceeds 4096 bytes");
        var request = JsonSerializer.Deserialize<Request>(File.ReadAllText(requestPath)) ?? throw new InvalidDataException("Missing request");
        if (!Regex.IsMatch(request.Nonce, "^[0-9a-f]{32}$") || Math.Abs((DateTimeOffset.UtcNow - request.CreatedUtc).TotalSeconds) > 30)
            throw new InvalidDataException("Invalid nonce or request age");
        if (seen.Contains(request.Nonce)) return;
        if (seen.Count >= 1000) throw new InvalidDataException("Probe request limit reached");
        seen.Add(request.Nonce);
        if (request.Operation == "snapshot") WriteNew(Path.Combine(config.OutputRoot, "snapshot-" + request.Nonce + ".json"), Capture(true));
        else if (request.Operation == "record") Start(request);
        else if (request.Operation == "arm_recording") ArmRecording(request);
        else if (request.Operation == "start_scene") StartScene(request);
        else if (request.Operation == "save_usda") SaveUsda(request);
        else if (request.Operation == "load_solution") LoadSolution(request);
        else throw new InvalidDataException("Unknown native recording operation");
    }

    private void RequireBenchmark()
    {
        if (!ProjectMatches() || !host.IsReady)
            throw new InvalidOperationException("The registered test project and program must be active");
        var current = SolutionPath();
        var binding = config!.AllowedSolutions.SingleOrDefault(scene =>
            Canonical(scene.Path).Equals(current, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidOperationException("The active solution is not registered for native execution");
        var programs = host.SolutionManager.GetRobotProgramsName().ToArray();
        var expected = Path.GetFileNameWithoutExtension(binding.ProgramName);
        if (programs.Length != 1 || Path.GetFileNameWithoutExtension(programs[0].Item2) != expected ||
            Path.GetFileNameWithoutExtension(host.GetProgramName()) != expected)
            throw new InvalidOperationException("Active program does not match the registered test program");
        var pieces = host.SolutionManager.GetPiecesInfo().ToArray();
        if (pieces.Length != 1 || pieces[0].Name != binding.PieceName)
            throw new InvalidOperationException("Active workpiece does not match the registered test workpiece");
    }

    private static (Type Contract, object Service) SolutionService()
    {
        var app = PublicType("Olpp.App");
        var provider = app == null ? null : PublicProperty(null, app, "ServiceProvider", true) as IServiceProvider;
        var contract = PublicType("Olpp.Core.Solutions.Services.SolutionService")
            ?? throw new InvalidOperationException("Public EPS solution service is unavailable");
        var service = provider?.GetService(contract) ?? throw new InvalidOperationException("EPS solution service is not registered");
        return (contract, service);
    }

    private static string? SolutionPath()
    {
        var (contract, service) = SolutionService();
        var path = PublicProperty(service, contract, "SolutionPath") as string;
        return string.IsNullOrWhiteSpace(path) || !Path.IsPathFullyQualified(path) ? null : Canonical(path);
    }

    private void LoadSolution(Request request)
    {
        if (!ProjectMatches()) throw new InvalidOperationException("The isolated EPS project is not active");
        var path = Canonical(request.SolutionPath);
        if (!Under(path, config!.AllowedProjectFolder) || !File.Exists(path) ||
            !config.AllowedSolutions.Any(scene => Canonical(scene.Path).Equals(path, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException("Only explicitly registered files under the isolated project can be loaded");
        var response = Path.Combine(config.OutputRoot, "load-" + request.Nonce + ".json");
        var model = SimulationViewModel();
        var (recorderContract, recorder) = RecordingService();
        var flags = recorderContract == null || recorder == null ? null : RecordingFlags(recorderContract, recorder);
        if (PublicProperty(model, model.GetType(), "SimulationRunning") is not false || flags == null ||
            flags["IsBusy"] != false || flags["IsRecording"] != false || flags["IsEnabled"] != false ||
            Application.Current.Windows.OfType<Window>().Any(window => window.IsVisible && window != Application.Current.MainWindow))
        {
            WriteNew(response, new { state = "blocked", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, reason = "EPS must be idle before changing the native solution", called = false });
            return;
        }
        try
        {
            var (contract, service) = SolutionService();
            var load = contract.GetMethod("LoadSolution", new[] { typeof(string) })
                ?? throw new MissingMethodException(contract.FullName, "LoadSolution");
            var check = contract.GetMethod("CheckIfFilesExists", new[] { typeof(string), typeof(string).MakeByRefType() });
            object?[] args = { path, null };
            if (check?.Invoke(service, args) is not true)
                throw new InvalidDataException("EPS rejected the solution dependencies: " + args[1]);
            load.Invoke(service, new object[] { path });
            var observed = SolutionPath();
            WriteNew(response, new { state = string.Equals(observed, path, StringComparison.OrdinalIgnoreCase) ? "loaded" : "unconfirmed",
                eps_pid = Environment.ProcessId, module_instance = instance, utc = DateTime.UtcNow, called = true, solution_path = observed });
        }
        catch (Exception ex)
        {
            WriteNew(response, new { state = "unconfirmed", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, error = Describe(ex) });
        }
    }

    private void ArmRecording(Request request)
    {
        RequireBenchmark();
        var (contract, recording) = RecordingService();
        if (contract == null || recording == null)
            throw new InvalidOperationException("EPS OpenUSD recording service is unavailable");
        var before = RecordingFlags(contract, recording);
        var receiptPath = Path.Combine(config!.OutputRoot, "arm-" + request.Nonce + ".json");
        if (before["IsAvailable"] != true || before["CanToggle"] != true || before["IsBusy"] != false ||
            before["IsEnabled"] != false || before["IsRecording"] != false)
        {
            WriteNew(receiptPath, new { state = "blocked", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, reason = "EPS did not report an idle, available OpenUSD recorder", before, called = false });
            return;
        }

        try
        {
            simulationDispatchedByBridge = false;
            var toggle = contract.GetMethod("ToggleRecording", Type.EmptyTypes)
                ?? throw new MissingMethodException(contract.FullName, "ToggleRecording");
            toggle.Invoke(recording, null);
            var after = RecordingFlags(contract, recording);
            recordingArmedByBridge = after["IsEnabled"] == true;
            WriteNew(receiptPath, new { state = after["IsEnabled"] == true ? "armed" : "unconfirmed",
                eps_pid = Environment.ProcessId, module_instance = instance, utc = DateTime.UtcNow, before, after, called = true });
        }
        catch (Exception ex)
        {
            WriteNew(receiptPath, new { state = "unconfirmed", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, before, called = true, error = Describe(ex) });
        }
    }

    private void StartScene(Request request)
    {
        RequireBenchmark();
        var (contract, recording) = RecordingService();
        var flags = contract == null || recording == null ? null : RecordingFlags(contract, recording);
        var viewModel = SimulationViewModel();
        var type = viewModel.GetType();
        var command = PublicProperty(viewModel, type, "StartSimulationCommand") as ICommand;
        var receiptPath = Path.Combine(config!.OutputRoot, "play-" + request.Nonce + ".json");
        if (!recordingArmedByBridge || flags == null || flags["IsAvailable"] != true || flags["IsEnabled"] != true ||
            flags["IsBusy"] != false || flags["IsRecording"] != false ||
            PublicProperty(viewModel, type, "PaintSimulationAvailable") is not true ||
            PublicProperty(viewModel, type, "PaintSimulationEnabled") is not true ||
            PublicProperty(viewModel, type, "SimulationRunning") is not false || command?.CanExecute(null) != true)
        {
            WriteNew(receiptPath, new { state = "blocked", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, reason = "The native Play or recording preconditions are unavailable", called = false });
            return;
        }

        // Save a dispatch receipt before the native command. A crash or timeout after
        // this point needs inspection of the same execution before any retry.
        WriteNew(Path.Combine(config.OutputRoot, "play-dispatched-" + request.Nonce + ".json"), new {
            eps_pid = Environment.ProcessId, module_instance = instance, utc = DateTime.UtcNow,
            program = host.GetProgramName(), project_folder = host.ProjectFolderPath });
        try
        {
            recordingArmedByBridge = false;
            simulationDispatchedByBridge = true;
            command.Execute(null);
            WriteNew(receiptPath, new { state = "requested", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, called = true });
        }
        catch (Exception ex)
        {
            WriteNew(receiptPath, new { state = "unconfirmed", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, called = true, error = Describe(ex) });
        }
    }

    private void SaveUsda(Request request)
    {
        RequireBenchmark();
        var path = Path.Combine(config!.OutputRoot, "save-" + request.Nonce + ".json");
        var modal = Application.Current.Windows.OfType<Window>().SingleOrDefault(window => window.IsVisible &&
            window.GetType().FullName == "Olpp.Infrastructure.OpenUSD.RecordingExportWindow");
        var (contract, recorder) = RecordingService();
        var flags = contract == null || recorder == null ? null : RecordingFlags(contract, recorder);
        var model = SimulationViewModel();
        if (!simulationDispatchedByBridge || modal == null || flags?["IsBusy"] != true ||
            flags["IsRecording"] != false || PublicProperty(model, model.GetType(), "SimulationRunning") is not false)
        {
            WriteNew(path, new { state = "blocked", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, reason = "No completed bridge-started EPS recording is waiting to be exported", called = false });
            return;
        }
        var controls = VisualChildren(modal).Take(2500).ToArray();
        var usd = controls.OfType<RadioButton>().SingleOrDefault(button => button.Name == "UsdOutput" && button.IsVisible);
        var save = controls.OfType<Button>().SingleOrDefault(button => button.Content is string label && label == "Save"
            && button.IsVisible && button.IsEnabled);
        if (usd?.IsChecked != true || save == null)
        {
            WriteNew(path, new { state = "blocked", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, reason = "The EPS export window does not have the expected OpenUSD option and Save button", called = false });
            return;
        }
        var invoke = new ButtonAutomationPeer(save).GetPattern(PatternInterface.Invoke) as IInvokeProvider
            ?? throw new InvalidOperationException("The EPS recording Save button cannot be invoked");
        simulationDispatchedByBridge = false;
        WriteNew(Path.Combine(config.OutputRoot, "save-dispatched-" + request.Nonce + ".json"), new {
            eps_pid = Environment.ProcessId, module_instance = instance, utc = DateTime.UtcNow, output = "OpenUSD" });
        try
        {
            invoke.Invoke();
            WriteNew(path, new { state = "requested", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, called = true });
        }
        catch (Exception ex)
        {
            WriteNew(path, new { state = "unconfirmed", eps_pid = Environment.ProcessId, module_instance = instance,
                utc = DateTime.UtcNow, called = true, error = Describe(ex) });
        }
    }

    private static IEnumerable<DependencyObject> VisualChildren(DependencyObject root)
    {
        var queue = new Queue<DependencyObject>();
        queue.Enqueue(root);
        while (queue.Count > 0)
        {
            var next = queue.Dequeue();
            yield return next;
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(next); i++)
                queue.Enqueue(VisualTreeHelper.GetChild(next, i));
        }
    }

    private static object SimulationViewModel()
    {
        var window = Application.Current.MainWindow ?? throw new InvalidOperationException("EPS main window is unavailable");
        var modelType = PublicType("Olpp.Features.ToolBar.SimulationBarViewModel")
            ?? throw new InvalidOperationException("EPS simulation view model is unavailable");
        return VisualChildren(window).Take(12000).OfType<FrameworkElement>()
            .Select(element => element.DataContext).Where(value => value != null && modelType.IsInstanceOfType(value))
            .Distinct(ReferenceEqualityComparer.Instance).Single();
    }

    private static object OpenWindows()
    {
        return Application.Current.Windows.OfType<Window>().Where(window => window.IsVisible && window != Application.Current.MainWindow)
            .Select(window => new {
                title = window.Title, type = window.GetType().FullName, context_type = window.DataContext?.GetType().FullName,
                buttons = VisualChildren(window).Take(2500).OfType<Button>().Where(button => button.IsVisible)
                    .Select(button => new { name = button.Name, label = button.Content as string, enabled = button.IsEnabled,
                        command = button.Command?.GetType().FullName }).ToArray(),
                choices = VisualChildren(window).Take(2500).OfType<RadioButton>().Where(button => button.IsVisible)
                    .Select(button => new { name = button.Name, label = button.Content as string, selected = button.IsChecked }).ToArray()
            }).ToArray();
    }
    private void Start(Request request)
    {
        if (trace != null) throw new InvalidOperationException("A recording is already active");
        if (request.DurationMs < 1000 || request.DurationMs > 30000 || request.IntervalMs < 50 || request.IntervalMs > 1000)
            throw new InvalidDataException("Invalid recording bounds");
        observationDir = Path.Combine(config!.OutputRoot, "observation-" + request.Nonce);
        if (Directory.Exists(observationDir)) throw new IOException("Observation directory already exists");
        Directory.CreateDirectory(observationDir);
        trace = new StreamWriter(new FileStream(Path.Combine(observationDir, "trace.jsonl"), FileMode.CreateNew, FileAccess.Write, FileShare.Read), new UTF8Encoding(false));
        while (caughtPaintingExceptions.TryDequeue(out _)) { }
        Interlocked.Exchange(ref exceptionCandidates, 0); exceptionEventsWritten = 0;
        Interlocked.Exchange(ref exceptionEventsDropped, 0);
        Interlocked.Exchange(ref exceptionObservations, 0);
        Interlocked.Exchange(ref exceptionObservationLimitDrops, 0);
        durationMs = request.DurationMs; intervalMs = request.IntervalMs; frames = 0; elapsed = Stopwatch.StartNew();
        trace.WriteLine(JsonSerializer.Serialize(new { kind = "recording_created", utc = DateTime.UtcNow, monotonic_ticks = Stopwatch.GetTimestamp(), eps_pid = Environment.ProcessId }));
        trace.Flush();
        Append("recording_started", true);
        observePaintingExceptions = true;
        nextFrame = Stopwatch.GetTimestamp();
    }
    private Dictionary<string, object?> Capture(bool publicState)
    {
        if (!ProjectMatches()) throw new InvalidOperationException("Active project differs from the configured isolated project");
        var data = new Dictionary<string, object?> {
            ["utc"] = DateTime.UtcNow, ["monotonic_ticks"] = Stopwatch.GetTimestamp(), ["monotonic_frequency"] = Stopwatch.Frequency,
            ["eps_pid"] = Environment.ProcessId, ["module_instance"] = instance, ["project_guard_matched"] = true,
            ["wall_elapsed_ms"] = elapsed?.Elapsed.TotalMilliseconds, ["probe_alone_proves_native_execution_or_deposition"] = false,
            ["tcp_source"] = "IHostingApplication scene RobotTcpFrame; identity fallback is possible; not Rte joint telemetry",
            ["controller_program_verified"] = false, ["gun_state"] = null, ["raw_deposition"] = null
        };
        var errors = new Dictionary<string, string>();
        void Read(string key, Func<object?> get) { try { data[key] = get(); } catch (Exception ex) { data[key] = null; errors[key] = Describe(ex); } }
        Read("host_is_ready", () => host.IsReady);
        Read("project_folder", () => host.ProjectFolderPath);
        if (publicState) Read("solution_path", SolutionPath);
        Read("editor_program_name", () => host.GetProgramName());
        Read("scene_tcp_matrix", () => {
            var matrix = host.GetRobotCurrentTransformation();
            if (matrix.GetLength(0) > 4 || matrix.GetLength(1) > 4) throw new InvalidDataException("Unexpected matrix size");
            var rows = new double[matrix.GetLength(0)][];
            for (int i = 0; i < rows.Length; i++) { rows[i] = new double[matrix.GetLength(1)]; for (int j = 0; j < rows[i].Length; j++) {
                if (!double.IsFinite(matrix[i, j])) throw new InvalidDataException("Nonfinite matrix"); rows[i][j] = matrix[i, j]; } }
            return rows;
        });
        if (publicState)
        {
            Read("solution_programs", () => host.SolutionManager.GetRobotProgramsName().Select(p => new { robot = p.Item1, program = p.Item2 }).ToArray());
            Read("solution_piece_names", () => host.SolutionManager.GetPiecesInfo().Take(1000).Select(p => p.Name).ToArray());
            Read("public_native_state", PublicNativeState);
            Read("recording_api", RecordingApi);
            Read("open_windows", OpenWindows);
            Read("simulation_toolbar", () => {
                var model = SimulationViewModel();
                return new[] { "SimulationRunning", "PaintSimulationAvailable", "PaintSimulationEnabled", "ElapsedTimeText" }
                    .ToDictionary(name => name, name => PublicProperty(model, model.GetType(), name));
            });
            data["caught_exception_observer"] = new {
                module_version = "1.6", active_only_during_explicit_recording = true,
                filter = "painting stack or shader/GLSL/OpenGL message",
                maximum_events = 32, maximum_observations = 512, exception_candidates = Volatile.Read(ref exceptionCandidates),
                events_written = exceptionEventsWritten, events_dropped = Volatile.Read(ref exceptionEventsDropped),
                exceptions_observed = Volatile.Read(ref exceptionObservations), observation_limit_drops = Volatile.Read(ref exceptionObservationLimitDrops),
                exception_observation_complete = Volatile.Read(ref exceptionObservationLimitDrops) == 0 && Volatile.Read(ref exceptionEventsDropped) == 0,
                no_event_is_not_proof_of_success = true
            };
        }
        data["read_errors"] = errors;
        last = JsonSerializer.Serialize(data, Json);
        if (display != null) display.Text = last;
        return data;
    }

    private static object RecordingApi()
    {
        var assemblies = AppDomain.CurrentDomain.GetAssemblies();
        var contracts = new[] {
            "Olpp.Infrastructure.OpenUSD.IOpenUsdRecordingService",
            "Olpp.Features.ToolBar.SimulationBarViewModel",
            "Olpp.Core.Solutions.Services.SolutionService",
            "Olpp.Infrastructure.FileServices.Application.ApplicationFileService",
            "VirtualRobot.IPaintProgramBuilder"
        };
        var types = contracts.Select(name => {
            var type = PublicType(name);
            return new {
                name, assembly = type?.Assembly.GetName().Name,
                methods = type?.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                    .Select(method => method.ToString()).OrderBy(signature => signature).ToArray(),
                properties = type?.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                    .Select(property => property.ToString()).OrderBy(signature => signature).ToArray()
            };
        }).ToArray();
        var (serviceType, recording) = RecordingService();
        return new {
            types, service_registered = recording != null,
            service_type = recording?.GetType().FullName,
            service_assembly = recording?.GetType().Assembly.GetName().Name,
            flags = recording == null || serviceType == null ? null : RecordingFlags(serviceType, recording),
            candidate_assemblies = assemblies.Select(a => a.GetName().Name)
                .Where(name => name != null && (name.StartsWith("Olpp.") || name.Contains("Usd", StringComparison.OrdinalIgnoreCase)))
                .OrderBy(name => name).ToArray()
        };
    }

    private static (Type?, object?) RecordingService()
    {
        var app = PublicType("Olpp.App");
        var provider = app == null ? null : PublicProperty(null, app, "ServiceProvider", true) as IServiceProvider;
        var contract = PublicType("Olpp.Infrastructure.OpenUSD.IOpenUsdRecordingService");
        return (contract, contract == null ? null : provider?.GetService(contract));
    }

    private static Dictionary<string, bool?> RecordingFlags(Type contract, object recording)
    {
        return new[] { "IsAvailable", "IsEnabled", "IsBusy", "IsRecording", "CanToggle" }
            .ToDictionary(name => name, name => contract.GetProperty(name)?.GetValue(recording) as bool?);
    }

    private static Type? PublicType(string name)
    {
        var matches = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(name, false)).Where(t => t != null && t.IsVisible).ToArray();
        if (matches.Length > 1) throw new InvalidOperationException("Ambiguous loaded public type: " + name);
        return matches.SingleOrDefault();
    }
    private static object? PublicProperty(object? target, Type type, string name, bool isStatic = false)
    {
        for (Type? current = type; current != null; current = current.BaseType)
        {
            if (!current.IsVisible) continue;
            var property = current.GetProperty(name, BindingFlags.Public | BindingFlags.DeclaredOnly | (isStatic ? BindingFlags.Static : BindingFlags.Instance));
            if (property == null || property.GetIndexParameters().Length != 0) continue;
            var getter = property.GetMethod;
            if (getter == null || !getter.IsPublic || getter.IsStatic != isStatic) continue;
            return property.GetValue(target);
        }
        throw new MissingMemberException(type.FullName, name);
    }
    private static object? PublicNativeState()
    {
        var app = PublicType("Olpp.App") ?? throw new InvalidOperationException("Loaded public Olpp.App not found");
        var provider = PublicProperty(null, app, "ServiceProvider", true) as IServiceProvider
            ?? throw new InvalidOperationException("Public application service provider unavailable");
        var result = new Dictionary<string, object?> { ["access"] = "public App.ServiceProvider / IServiceProvider / public getters only",
            ["service_resolution_may_instantiate_registered_services"] = true, ["private_reflection_used"] = false,
            ["availability_setters_called"] = false, ["simulator_or_controller_commands_called"] = false };
        foreach (var entry in new[] { ("paint", "Olpp.Core.Painting.PaintSimulationService"), ("viewport_manager", "Olpp.Infrastructure.Eyeshot.Viewport.EyeshotManager") })
        {
            var type = PublicType(entry.Item2);
            var first = type == null ? null : provider.GetService(type);
            var second = type == null ? null : provider.GetService(type);
            bool stable = first != null && ReferenceEquals(first, second);
            result[entry.Item1 + "_service_present"] = first != null;
            result[entry.Item1 + "_same_instance_on_repeated_resolution"] = stable;
            if (!stable) continue;
            if (entry.Item1 == "paint") {
                result["paint_enabled"] = PublicProperty(first, type!, "Enabled");
                result["paint_runtime_available"] = PublicProperty(first, type!, "RuntimeAvailable");
            } else result["manager_viewport_present"] = PublicProperty(first, type!, "Viewport") != null;
        }
        result["public_shader_catalog"] = PublicShaderCatalogObservation.Capture(provider);
        // Traverse the current application's public WPF visual tree, never the internal Viewport.design field.
        var designs = new List<object>(); var queue = new Queue<DependencyObject>();
        if (Application.Current?.MainWindow != null) queue.Enqueue(Application.Current.MainWindow);
        int visited = 0;
        while (queue.Count > 0 && visited++ < 2000)
        {
            var node = queue.Dequeue(); bool design = false;
            for (Type? t = node.GetType(); t != null; t = t.BaseType) if (t.FullName == "devDept.Eyeshot.Control.WorkspaceBase") design = true;
            if (design && node is FrameworkElement element && designs.Count < 4)
            {
                var state = new Dictionary<string, object?> { ["type"] = node.GetType().FullName, ["is_loaded"] = element.IsLoaded,
                    ["is_visible"] = element.IsVisible, ["width"] = element.ActualWidth, ["height"] = element.ActualHeight };
                try {
                    var context = PublicProperty(node, node.GetType(), "RenderContext");
                    state["render_context_present"] = context != null;
                    state["render_context_type"] = context?.GetType().FullName;
                    if (context != null) {
                        // Fixed public getters only. Values belong to this EPS viewport, not a new GL context.
                        foreach (var name in new[] { "Vendor", "VendorName", "RendererVersion", "RendererName" }) {
                            try {
                                var value = PublicProperty(context, context.GetType(), name);
                                var rendered = value?.ToString();
                                state["context_" + name] = rendered == null ? null : rendered.Substring(0, Math.Min(rendered.Length, 512));
                            } catch (Exception ex) { state["context_" + name + "_error"] = Describe(ex); }
                        }
                    }
                }
                catch (Exception ex) { state["render_context_error"] = Describe(ex); }
                try { var renderer = PublicProperty(node, node.GetType(), "Renderer"); state["renderer"] = renderer is Enum ? renderer.ToString() : renderer?.GetType().FullName; }
                catch (Exception ex) { state["renderer_error"] = Describe(ex); }
                designs.Add(state);
            }
            try { int count = VisualTreeHelper.GetChildrenCount(node); for (int i = 0; i < count; i++) queue.Enqueue(VisualTreeHelper.GetChild(node, i)); }
            catch { /* Non-visual content is not queried through private access. */ }
        }
        result["public_visual_tree_nodes_visited"] = visited;
        result["public_visual_tree_truncated"] = queue.Count > 0;
        result["shader_public_type_members"] = new[] { "Olpp.Infrastructure.Eyeshot.Painting.FileShaderSourceProvider", "Olpp.Infrastructure.Eyeshot.Painting.OffscreenPaintingSimulationRuntime", "Olpp.Core.Painting.PaintSimulationService", "devDept.Graphics.IShader" }.Select(name => { var t = PublicType(name); return new { type = name, visible = t != null, public_members = t == null ? Array.Empty<string>() : t.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static).Select(m => m.MemberType + ":" + m.Name).Distinct().Take(60).ToArray() }; }).ToArray();
        result["design_controls"] = designs;
        return result;
    }
    private void Append(string kind, bool publicState)
    {
        if (trace == null) return;
        var data = Capture(publicState); data["kind"] = kind;
        trace.WriteLine(JsonSerializer.Serialize(data, Json)); trace.Flush(); frames++;
        if (!stopping && trace.BaseStream.Position > 1024 * 1024) Stop("one_mib_limit");
    }
    private void Event(string kind)
    {
        var dispatcher = Application.Current?.Dispatcher;
        if (dispatcher != null && !dispatcher.HasShutdownStarted) dispatcher.BeginInvoke(new Action(() => Guard(() => { if (trace != null && ProjectMatches()) Append(kind, false); })));
    }
    private void Stop(string reason)
    {
        if (trace == null || stopping) return;
        stopping = true;
        observePaintingExceptions = false;
        string? error = null;
        try { DrainCaughtPaintingExceptions(); if (ProjectMatches()) Append("recording_stopped:" + reason, true); }
        catch (Exception ex) { error = Describe(ex); }
        finally { try { trace?.Dispose(); } catch (Exception ex) { error ??= Describe(ex); } trace = null; elapsed?.Stop(); stopping = false; }
        try { var path = Path.Combine(observationDir!, "trace.jsonl"); WriteNew(Path.Combine(observationDir!, "receipt.json"), new {
            reason, utc = DateTime.UtcNow, frames, wall_elapsed_ms = elapsed?.Elapsed.TotalMilliseconds, bytes = new FileInfo(path).Length,
            sha256 = Hash(path), finalization_error = error,
            caught_exception_observer = new {
                filter = "painting stack or shader/GLSL/OpenGL message",
                exception_candidates = Volatile.Read(ref exceptionCandidates),
                events_written = exceptionEventsWritten, events_dropped = Volatile.Read(ref exceptionEventsDropped),
                exceptions_observed = Volatile.Read(ref exceptionObservations), observation_limit_drops = Volatile.Read(ref exceptionObservationLimitDrops),
                exception_observation_complete = Volatile.Read(ref exceptionObservationLimitDrops) == 0,
                concurrent_event_completion_at_stop_possible = true,
                no_event_is_not_proof_of_success = true
            }, native_execution_or_deposition_verified = false }); }
        catch (Exception ex) { last = Describe(ex); }
    }
    private static string Describe(Exception ex) => ex.GetType().Name + ": " + ex.Message;
    private static string Hash(string path) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant();
    private static void WriteNew(string path, object value)
    {
        var bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(value, Json) + "\n");
        using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        file.Write(bytes);
    }
    private static string Canonical(string value)
    {
        if (!Path.IsPathFullyQualified(value)) throw new InvalidDataException("Installer paths must be absolute");
        var full = Path.GetFullPath(value).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (full.Length < 4) throw new InvalidDataException("A task directory is required");
        return full;
    }
    private static bool Under(string path, string root) => path.Equals(root, StringComparison.OrdinalIgnoreCase)
        || path.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase);
    public sealed class Config
    {
        public string AllowedWorkRoot { get; set; } = "";
        public string AllowedProjectFolder { get; set; } = "";
        public SceneBinding[] AllowedSolutions { get; set; } = Array.Empty<SceneBinding>();
        public string OutputRoot { get; set; } = "";
    }
    public sealed class SceneBinding
    {
        public string Path { get; set; } = "";
        public string ProgramName { get; set; } = "";
        public string PieceName { get; set; } = "";
    }
    public sealed class Request
    {
        public string Nonce { get; set; } = "";
        public string Operation { get; set; } = "";
        public string SolutionPath { get; set; } = "";
        public DateTimeOffset CreatedUtc { get; set; }
        public int DurationMs { get; set; } = 12000;
        public int IntervalMs { get; set; } = 100;
    }
}

param(
    [Parameter(Mandatory=$true)][string]$CompilerDll,
    [Parameter(Mandatory=$true)][string]$CoreRuntimeDirectory,
    [Parameter(Mandatory=$true)][string]$WindowsDesktopRuntimeDirectory,
    [Parameter(Mandatory=$true)][string]$EpsContractPath,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'build-direct')
)
$ErrorActionPreference = 'Stop'

function Resolve-Existing([string]$Value) {
    $full = (Resolve-Path -LiteralPath $Value -ErrorAction Stop).ProviderPath
    if ($full.Contains('"') -or $full.Contains("`n") -or $full.Contains("`r")) {
        throw 'Unexpected path characters in compiler response file input'
    }
    return $full
}
function Receipt([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    return @{ path=$item.FullName; size=$item.Length; sha256=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
}
function Save-Json([string]$Path, $Value) {
    $json = ConvertTo-Json -InputObject $Value -Depth 12
    [IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

$compiler = Resolve-Existing $CompilerDll
$core = Resolve-Existing $CoreRuntimeDirectory
$desktop = Resolve-Existing $WindowsDesktopRuntimeDirectory
$contract = Resolve-Existing $EpsContractPath
$source = Resolve-Existing (Join-Path $PSScriptRoot 'NativeStateProbe.cs')
$output = [IO.Path]::GetFullPath($OutputDirectory)
if ($output.Contains('"') -or $output.Contains("`n") -or $output.Contains("`r")) { throw 'Invalid output directory' }
if (Test-Path -LiteralPath $output) { throw 'Use a NEW output directory; existing results are preserved' }
New-Item -ItemType Directory -Path $output | Out-Null

try {
    Get-Command dotnet -ErrorAction Stop | Out-Null
    # The desktop framework replaces same-named core facades such as WindowsBase.
    $managed = @{}
    $excluded = @()
    foreach ($directory in @($core, $desktop)) {
        foreach ($file in Get-ChildItem -LiteralPath $directory -Filter '*.dll' -File) {
            try {
                [Reflection.AssemblyName]::GetAssemblyName($file.FullName) | Out-Null
                $managed[$file.Name] = $file.FullName
            } catch {
                $excluded += @{ path=$file.FullName; reason=$_.Exception.GetType().FullName }
            }
        }
    }
    $managed['Olpp.Module.dll'] = $contract
    foreach ($required in @('System.Private.CoreLib.dll','System.Runtime.dll','System.Text.Json.dll','WindowsBase.dll','PresentationCore.dll','PresentationFramework.dll','Olpp.Module.dll')) {
        if (-not $managed.ContainsKey($required)) { throw "Missing existing managed reference: $required" }
    }
    $dll = Join-Path $output 'SprayBot.EpsNativeStateProbe.dll'
    $response = Join-Path $output 'compile.rsp'
    $lines = @('/nologo','/nostdlib+','/target:library','/langversion:latest','/nullable:enable','/deterministic+',('/out:"' + $dll + '"'))
    $refs = @()
    foreach ($name in @($managed.Keys | Sort-Object)) {
        $path = Resolve-Existing $managed[$name]
        $lines += '/r:"' + $path + '"'
        $refs += Receipt $path
    }
    $lines += '"' + $source + '"'
    foreach ($helper in @('NativeGlObservation.cs','PublicShaderCatalogObservation.cs')) { $lines += '"' + (Resolve-Existing (Join-Path $PSScriptRoot $helper)) + '"' }
    [IO.File]::WriteAllLines($response, $lines, (New-Object Text.UTF8Encoding($false)))
    $priorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $rawOutput = @(& dotnet $compiler /noconfig "@$response" 2>&1); $compilerExit = $LASTEXITCODE }
    finally { $ErrorActionPreference = $priorActionPreference }
    $text = ($rawOutput | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    $bytes = [Text.Encoding]::UTF8.GetBytes($text)
    $hash = [Security.Cryptography.SHA256]::Create()
    try { $outputHash = ([BitConverter]::ToString($hash.ComputeHash($bytes))).Replace('-','').ToLowerInvariant() }
    finally { $hash.Dispose() }
    $compilerOutput = @{ size=$bytes.Length; sha256=$outputHash; file=$null }
    if ($bytes.Length -gt 0) {
        $log = Join-Path $output 'compiler-output.txt'
        [IO.File]::WriteAllBytes($log, $bytes)
        $compilerOutput.file = $log
    }
    # Empty compiler stdout/stderr is truthfully recorded, never a flat zero-byte attachment.
    $result = @{ at_utc=[DateTime]::UtcNow.ToString('o'); exit_code=$compilerExit;
        compiler=(Receipt $compiler); source=(Receipt $source); contract=(Receipt $contract);
        core_runtime=$core; desktop_runtime=$desktop; references=$refs; excluded_unmanaged_candidates=$excluded;
        compiler_output=$compilerOutput; output_dll=$null; loaded_in_eps=$false;
        scope='Compilation only. No deployment, vendor file changes, native calls, or controller operations.' }
    if ($compilerExit -eq 0 -and (Test-Path -LiteralPath $dll)) { $result.output_dll = Receipt $dll }
    Save-Json (Join-Path $output 'compile-receipt.json') $result
    if ($compilerExit -ne 0) { throw "Compiler exit $compilerExit; inspect compile-receipt.json and nonempty compiler output" }
    if (-not $result.output_dll) { throw 'Compiler reported success without an output DLL' }
    Write-Output (ConvertTo-Json -InputObject @{ exit_code=$compilerExit; output_dll=$result.output_dll; loaded_in_eps=$false } -Depth 4)
} catch {
    Save-Json (Join-Path $output 'build-failure.json') @{ at_utc=[DateTime]::UtcNow.ToString('o'); error=$_.Exception.Message; loaded_in_eps=$false }
    throw
}

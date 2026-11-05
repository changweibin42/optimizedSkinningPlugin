# ================================================================
# build.ps1  --  Build optimizedSkinningPlugin.mll via CMake
#
# Usage:
#   cd D:\visualStudioProject\optimizedSkinningPlugin
#   .\build.ps1              # incremental
#   .\build.ps1 -Clean       # wipe build dir first
#   .\build.ps1 -Config Debug
# ================================================================

param(
    [string]$Config = "Release",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# ---------------- paths ----------------
$ProjectDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$BuildDir   = Join-Path $ProjectDir "build"
$OutDll     = Join-Path $BuildDir "$Config\optimizedSkinningPlugin.mll"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Build optimizedSkinningPlugin ($Config)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Project : $ProjectDir"
Write-Host " Build   : $BuildDir"
Write-Host ""

# ---------------- env check ----------------
if (-not $env:DEVKIT_LOCATION) {
    Write-Host "[ERROR] DEVKIT_LOCATION is not set" -ForegroundColor Red
    Write-Host '        e.g.  $env:DEVKIT_LOCATION = "C:\Program Files\Autodesk\Maya2024\devkit"' -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $env:DEVKIT_LOCATION)) {
    Write-Host "[ERROR] DEVKIT_LOCATION path not found: $env:DEVKIT_LOCATION" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] DEVKIT_LOCATION = $env:DEVKIT_LOCATION" -ForegroundColor Green

# ---------------- cmake check ----------------
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "[ERROR] cmake not found in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] cmake: $($cmake.Source)" -ForegroundColor Green

# ---------------- clean ----------------
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "[OK] Removing old build dir" -ForegroundColor Green
    Remove-Item $BuildDir -Recurse -Force
}

# ---------------- configure ----------------
if (-not (Test-Path (Join-Path $BuildDir "CMakeCache.txt"))) {
    Write-Host ""
    Write-Host "[CMAKE] Configure..." -ForegroundColor Cyan
    & cmake -S $ProjectDir -B $BuildDir `
        -G "Visual Studio 17 2022" -A x64 `
        -DCMAKE_CONFIGURATION_TYPES=$Config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] cmake configure failed" -ForegroundColor Red
        exit 1
    }
}

# ---------------- build ----------------
Write-Host ""
Write-Host "[CMAKE] Build..." -ForegroundColor Cyan
& cmake --build $BuildDir --config $Config
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] build failed" -ForegroundColor Red
    exit 1
}

# ---------------- result ----------------
Write-Host ""
if (Test-Path $OutDll) {
    $info = Get-Item $OutDll
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host " BUILD SUCCEEDED" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host " Output : $($info.FullName)"
    Write-Host " Size   : $([math]::Round($info.Length/1KB,1)) KB"
    Write-Host " Time   : $($info.LastWriteTime)"
    Write-Host ""
    Write-Host " In Maya:"
    $dllPath = $info.FullName -replace '\\','/'
    Write-Host "   unloadPlugin `"optimizedSkinningPlugin`";"
    Write-Host "   loadPlugin `"$dllPath`";"
} else {
    Write-Host "[WARN] dll not found: $OutDll" -ForegroundColor Yellow
    Write-Host "       Check build output path" -ForegroundColor Yellow
}
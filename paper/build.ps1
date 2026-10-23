param(
    [switch]$Watch
)

$ErrorActionPreference = "Stop"
$paperRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$localTypst = Get-ChildItem -Path (Join-Path $paperRoot ".tools\typst") `
    -Filter "typst.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
$typst = Get-Command typst -ErrorAction SilentlyContinue

if ($null -ne $typst) {
    $typstCommand = $typst.Source
} elseif ($null -ne $localTypst) {
    $typstCommand = $localTypst.FullName
} else {
    throw "Typst was not found. Install it on PATH or place typst.exe at paper/.tools/typst/."
}

$arguments = @("compile", "--root", $paperRoot, "main.typ", "main.pdf")
if ($Watch) {
    $arguments[0] = "watch"
}

Push-Location $paperRoot
try {
    & $typstCommand @arguments
} finally {
    Pop-Location
}

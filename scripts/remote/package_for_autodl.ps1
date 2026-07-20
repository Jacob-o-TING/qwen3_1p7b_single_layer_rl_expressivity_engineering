param(
  [string]$Out = "qwen3_1p7b_single_layer_rl_expressivity_engineering_autodl.tar"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$parent = Split-Path -Parent $root
$name = Split-Path -Leaf $root
$outPath = Join-Path (Get-Location) $Out

Push-Location $parent
try {
  if (Test-Path -LiteralPath $outPath) {
    Remove-Item -LiteralPath $outPath -Force
  }
  tar --exclude="$name/runs" `
      --exclude="$name/data" `
      --exclude="$name/checkpoints" `
      --exclude="$name/__pycache__" `
      -cf "$outPath" "$name"
  Write-Host "Wrote package: $outPath"
} finally {
  Pop-Location
}

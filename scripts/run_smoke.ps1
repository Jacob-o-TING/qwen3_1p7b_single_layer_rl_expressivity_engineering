$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "python"
try {
  & python --version | Out-Null
  if ($LASTEXITCODE -ne 0) { $python = "py" }
} catch {
  $python = "py"
}
Push-Location $root
try {
  if ($python -eq "py") {
    py -3 -m qwen_single_layer_rl.training.dry_run --config configs/smoke_tiny.yaml --out runs/smoke
    py -3 -m unittest discover -s tests
  } else {
    python -m qwen_single_layer_rl.training.dry_run --config configs/smoke_tiny.yaml --out runs/smoke
    python -m unittest discover -s tests
  }
} finally {
  Pop-Location
}

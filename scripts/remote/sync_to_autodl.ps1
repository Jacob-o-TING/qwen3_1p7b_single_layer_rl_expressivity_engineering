param(
  [Parameter(Mandatory=$true)][string]$HostName,
  [Parameter(Mandatory=$true)][int]$Port,
  [string]$User = "root",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_autodl_codex",
  [string]$RemoteDir = "/root/autodl-tmp/qwen3_1p7b_single_layer_rl_expressivity_engineering",
  [string]$Package = "qwen3_1p7b_single_layer_rl_expressivity_engineering_autodl.tar"
)

$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$packagePath = Join-Path (Get-Location) $Package

if (-not (Test-Path -LiteralPath $packagePath)) {
  & (Join-Path $scriptDir "package_for_autodl.ps1") -Out $Package
}

$remote = "${User}@${HostName}"
$remoteTar = "/root/autodl-tmp/$Package"

ssh -i "$KeyPath" -p $Port $remote "mkdir -p /root/autodl-tmp"
scp -i "$KeyPath" -P $Port "$packagePath" "${remote}:${remoteTar}"
ssh -i "$KeyPath" -p $Port $remote "set -e; mkdir -p '$RemoteDir'; tar -xf '$remoteTar' -C /root/autodl-tmp; if [ -d /root/autodl-tmp/qwen3_1p7b_single_layer_rl_expressivity_engineering ]; then rsync -a --delete /root/autodl-tmp/qwen3_1p7b_single_layer_rl_expressivity_engineering/ '$RemoteDir'/; fi"

Write-Host "Synced to ${remote}:${RemoteDir}"

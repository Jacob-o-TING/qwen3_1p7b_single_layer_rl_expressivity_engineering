param(
  [Parameter(Mandatory=$true)][string]$HostName,
  [Parameter(Mandatory=$true)][int]$Port,
  [Parameter(Mandatory=$true)][string]$RunId,
  [string]$User = "root",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_autodl_codex",
  [string]$RemoteDir = "/root/autodl-tmp/qwen3_1p7b_single_layer_rl_expressivity_engineering",
  [string]$LocalRoot = "remote_artifacts"
)

$ErrorActionPreference = "Stop"

$remote = "${User}@${HostName}"
$localDir = Join-Path $LocalRoot $RunId
New-Item -ItemType Directory -Force $localDir | Out-Null

$remoteBundle = "/root/autodl-tmp/${RunId}_records.tar.gz"
$remoteRun = "${RemoteDir}/runs/${RunId}"
$remoteLogs = "${RemoteDir}/logs/${RunId}"

ssh -i "$KeyPath" -p $Port $remote "set -e; tar -czf '$remoteBundle' -C '$RemoteDir' 'runs/$RunId' 'logs/$RunId' 2>/dev/null || tar -czf '$remoteBundle' -C '$RemoteDir' 'logs/$RunId'"
scp -i "$KeyPath" -P $Port "${remote}:${remoteBundle}" "$localDir\"

Write-Host "Pulled records bundle to $localDir"
Write-Host "Remote run path was $remoteRun"
Write-Host "Remote log path was $remoteLogs"

# Multilingual Fake News Detector - deploy.ps1
# Push the current branch to a HuggingFace Space.
#
# Usage:
#   .\deploy.ps1 -SpaceOwner <hf-username> [-SpaceName multilingual-fake-news-detector] [-Force]
#
# Prerequisites:
#   1. pip install huggingface_hub
#   2. huggingface-cli login
#   3. Create the Space at https://huggingface.co/new-space (SDK: Gradio)

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceOwner,

    [string]$SpaceName = "multilingual-fake-news-detector",

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Assert-Command {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "Required command '$Name' was not found on PATH. $InstallHint"
        exit 1
    }
}

Assert-Command -Name "git" -InstallHint "Install Git from https://git-scm.com/."
Assert-Command -Name "huggingface-cli" -InstallHint "Install with: pip install huggingface_hub"

Write-Host "Checking HuggingFace login status..."
$whoami = & huggingface-cli whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Not logged into HuggingFace. Run: huggingface-cli login"
    exit 1
}
Write-Host "Logged in as: $whoami"

$remoteUrl = "https://huggingface.co/spaces/$SpaceOwner/$SpaceName"
$existing = & git remote get-url hf 2>$null
if ($LASTEXITCODE -eq 0) {
    if ($existing -ne $remoteUrl) {
        Write-Warning "Remote 'hf' already exists and points to: $existing"
        Write-Warning "Will overwrite to: $remoteUrl"
        $confirm = Read-Host "Continue? (y/N)"
        if ($confirm -ne "y") { Write-Host "Aborted."; exit 0 }
        & git remote set-url hf $remoteUrl
        if ($LASTEXITCODE -ne 0) { Write-Error "git remote set-url failed."; exit 1 }
    }
} else {
    Write-Host "Adding remote 'hf' -> $remoteUrl"
    & git remote add hf $remoteUrl
    if ($LASTEXITCODE -ne 0) { Write-Error "git remote add failed."; exit 1 }
}

$currentBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
Write-Host "Pushing branch '$currentBranch' to Space '$SpaceOwner/$SpaceName'..."

if ($Force) {
    & git push hf "${currentBranch}:main" --force
} else {
    & git push hf "${currentBranch}:main"
}

if ($LASTEXITCODE -ne 0) {
    Write-Error "git push failed. If this is the first push to an existing Space with unrelated history, re-run with -Force."
    exit 1
}

Write-Host ""
Write-Host "Deployed."
Write-Host "Space URL: $remoteUrl"

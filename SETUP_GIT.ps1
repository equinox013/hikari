<#
.SYNOPSIS
    One-shot cleanup + git initialisation for hikari_himari_machi.
    Run from PowerShell inside D:\Git Repos\hikari_himari_machi
    Usage: .\SETUP_GIT.ps1
#>

$Root = $PSScriptRoot

Write-Host "`n[1/4] Removing junk files..." -ForegroundColor Cyan

# Duplicate root-level model files (kept correctly in keras-pkl/)
Remove-Item -Force -ErrorAction SilentlyContinue "$Root\hikari_v2.keras"
Remove-Item -Force -ErrorAction SilentlyContinue "$Root\machi_v2.pkl"
Remove-Item -Force -ErrorAction SilentlyContinue "$Root\tokenizer.pkl"

# Draft/copy notebook
Remove-Item -Force -ErrorAction SilentlyContinue "$Root\Complaint Driver Preprocessing Script - HIMARI v2_UAT-Copy1.ipynb"

# Anaconda project metadata
Remove-Item -Force -Recurse -ErrorAction SilentlyContinue "$Root\anaconda_projects"

# Old nested git repo (we're initialising at root level)
Remove-Item -Force -Recurse -ErrorAction SilentlyContinue "$Root\hikari"

Write-Host "[2/4] Initialising git repository at root..." -ForegroundColor Cyan
Set-Location $Root
git init -b main

Write-Host "[3/4] Setting remote origin..." -ForegroundColor Cyan
git remote add origin https://github.com/equinox013/hikari.git

Write-Host "[4/4] Staging all files..." -ForegroundColor Cyan
git add .
git status

Write-Host "`nReady to commit. Run:" -ForegroundColor Green
Write-Host '  git commit -m "feat: add complaint driver classification pipeline (MACHI · HIMARI · HIKARI)"' -ForegroundColor Yellow
Write-Host '  git push --force origin main' -ForegroundColor Yellow
Write-Host "`nNote: --force is safe here. The remote only has the empty initial commit." -ForegroundColor DarkGray

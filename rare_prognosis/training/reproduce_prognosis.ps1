[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Models,
    [string]$CaseRoot = "",
    [string]$Config = "",
    [int]$NumCases = 10,
    [int]$TrainCount = 8,
    [int]$Seed = 42,
    [string]$TrainIds = "",
    [string]$TestIds = "",
    [string]$OutputRoot = "",
    [string]$Python = "",
    [int]$NumWorkers = 4,
    [int]$CvFolds = 5,
    [ValidateSet("overall_outcome", "functional_status", "symptom_burden", "all")]
    [string]$Task = "all"
)

<#
.SYNOPSIS
Runs the end-to-end RarePrognosis code demo on a deterministic subset of data_500.

.EXAMPLE
.\rare_prognosis\training\reproduce_prognosis.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Models GPT-5,o3-mini
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
if (-not $CaseRoot) { $CaseRoot = Join-Path $RepoRoot "data_500" }
if (-not $Config) { $Config = Join-Path $RepoRoot "llm_config.json" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "outputs\prognosis_demo" }
if (-not $Python) {
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
}

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Read-CaseIds {
    param([string]$Path)
    $value = Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    return @($value | ForEach-Object { [string]$_ })
}

function Write-CaseIds {
    param([string]$Path, [string[]]$Ids)
    $json = ConvertTo-Json -InputObject @($Ids) -Depth 3
    [System.IO.File]::WriteAllText($Path, $json, [System.Text.UTF8Encoding]::new($false))
}

function Test-EligibleCase {
    param([string]$CaseId)
    $CasePath = Join-Path $CaseRoot $CaseId
    $Required = @("prognosis_prediction.json", "prognosis_new.json")
    return (Test-Path -LiteralPath $CasePath -PathType Container) -and
        (@($Required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $CasePath $_) -PathType Leaf) }).Count -eq 0)
}

if (-not (Test-Path -LiteralPath $CaseRoot -PathType Container)) {
    throw "Case root does not exist: $CaseRoot"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "LLM config does not exist: $Config"
}
if (($TrainIds -and -not $TestIds) -or ($TestIds -and -not $TrainIds)) {
    throw "-TrainIds and -TestIds must be provided together."
}

$ModelsCsv = $Models -join ','
$ModelTags = @($ModelsCsv.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($ModelTags.Count -eq 0) { throw "-Models must contain at least one model tag." }
$ConfigEntries = Get-Content -LiteralPath $Config -Raw -Encoding utf8 | ConvertFrom-Json
$ConfiguredNames = @()
foreach ($ConfigEntry in @($ConfigEntries)) {
    if ($null -ne $ConfigEntry.model) { $ConfiguredNames += [string]$ConfigEntry.model }
    if ($null -ne $ConfigEntry.tags) {
        $ConfiguredNames += @($ConfigEntry.tags | ForEach-Object { [string]$_ })
    }
}
$MissingModels = @($ModelTags | Where-Object { $_ -notin $ConfiguredNames })
if ($MissingModels.Count -gt 0) {
    throw "Models absent from config: $($MissingModels -join ', ')"
}

Invoke-Python @("-c", "import numpy,openai,sklearn; print('Python dependencies ready.')")

$SplitRoot = Join-Path $OutputRoot "splits"
$LlmRoot = Join-Path $OutputRoot "llm_outputs"
$FeatureRoot = Join-Path $OutputRoot "features"
$ModelRoot = Join-Path $OutputRoot "model"
$ResultRoot = Join-Path $OutputRoot "results"
@($SplitRoot, $LlmRoot, $FeatureRoot, $ModelRoot, $ResultRoot) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

if (-not $TrainIds) {
    $AutoSplitCode = "import json,random,sys; from pathlib import Path; root=Path(sys.argv[1]); n=int(sys.argv[2]); nt=int(sys.argv[3]); seed=int(sys.argv[4]); out=Path(sys.argv[5]); req=['prognosis_prediction.json','prognosis_new.json']; ids=sorted(p.name for p in root.iterdir() if p.is_dir() and all((p/f).is_file() for f in req)); assert 2<=nt<n<=len(ids),f'require 2 <= train-count < num-cases <= {len(ids)}'; random.Random(seed).shuffle(ids); chosen=ids[:n]; out.mkdir(parents=True,exist_ok=True); [(out/name).write_text(json.dumps(vals,indent=2),encoding='utf-8') for name,vals in [('all.json',chosen),('train.json',chosen[:nt]),('test.json',chosen[nt:])]]"
    Invoke-Python @("-c", $AutoSplitCode, $CaseRoot, $NumCases, $TrainCount, $Seed, $SplitRoot)
    $TrainCaseIds = Read-CaseIds (Join-Path $SplitRoot "train.json")
    $TestCaseIds = Read-CaseIds (Join-Path $SplitRoot "test.json")
} else {
    $TrainCaseIds = Read-CaseIds $TrainIds
    $TestCaseIds = Read-CaseIds $TestIds
    $Overlap = @($TrainCaseIds | Where-Object { $_ -in $TestCaseIds })
    if ($TrainCaseIds.Count -lt 2) { throw "At least two training cases are required." }
    if ($TestCaseIds.Count -eq 0) { throw "Test split must not be empty." }
    if ($Overlap.Count -gt 0) { throw "Train/test overlap: $($Overlap -join ', ')" }
}

$TrainIds = Join-Path $SplitRoot "train.json"
$TestIds = Join-Path $SplitRoot "test.json"
$AllIds = Join-Path $SplitRoot "all.json"
$AllCaseIds = @($TrainCaseIds + $TestCaseIds)
Write-CaseIds $TrainIds $TrainCaseIds
Write-CaseIds $TestIds $TestCaseIds
Write-CaseIds $AllIds $AllCaseIds

$InvalidIds = @($AllCaseIds | Where-Object { -not (Test-EligibleCase $_) })
if ($InvalidIds.Count -gt 0) {
    throw "Ineligible or missing cases: $($InvalidIds -join ', ')"
}

Write-Host "RarePrognosis code demo"
Write-Host "  cases:       $AllIds"
Write-Host "  train/test:  $TrainIds / $TestIds"
Write-Host "  models:      $($ModelTags -join ',')"
Write-Host "  output:      $OutputRoot"

foreach ($ModelTag in $ModelTags) {
    Invoke-Python @(
        "-m", "rare_prognosis.training.generate_llm_outputs", $CaseRoot, $LlmRoot,
        "--model", $ModelTag, "--config", $Config, "--case-ids", $AllIds,
        "--num-workers", $NumWorkers
    )
}

Invoke-Python @(
    "-m", "rare_prognosis.training.prepare_data",
    "--case-root", $CaseRoot, "--llm-root", $LlmRoot,
    "--result-root", $ResultRoot, "--train-ids", $TrainIds, "--test-ids", $TestIds
)
Invoke-Python @(
    "-m", "rare_prognosis.training.build_features",
    "--results-root", $ResultRoot, "--models-root", $LlmRoot,
    "--train-ids", $TrainIds, "--test-ids", $TestIds,
    "--out-dir", $FeatureRoot, "--task", $Task
)
Invoke-Python @(
    "-m", "rare_prognosis.training.train_models",
    "--features-dir", $FeatureRoot, "--out-dir", $ModelRoot,
    "--task", $Task, "--seed", $Seed, "--cv-folds", $CvFolds
)
Invoke-Python @(
    "-m", "rare_prognosis.training.infer_models",
    "--results-root", $ResultRoot, "--models-root", $LlmRoot,
    "--train-ids", $TrainIds, "--test-ids", $TestIds,
    "--models-dir", $ModelRoot, "--task", $Task
)

Write-Host "Demo complete: $OutputRoot"

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Models,

    [ValidateSet("primary", "followup")]
    [string]$VisitType = "primary",

    [string]$CaseRoot = "",
    [string]$Config = "",
    [int]$NumCases = 10,
    [int]$TrainCount = 8,
    [int]$Seed = 42,
    [string]$TrainIds = "",
    [string]$TestIds = "",
    [string]$OutputRoot = "",
    [string]$ScoreRoot = "",
    [string]$Python = "",
    [int]$NumWorkers = 4,
    [int]$FeatureWorkers = 2,
    [int]$FeatureGpus = 0,
    [switch]$NoRag,
    [switch]$UseGpu
)

<#
.SYNOPSIS
Runs the end-to-end RareDiagnosis code demo on Windows PowerShell.

.EXAMPLE
.\rare_diagnosis\training\reproduce_diag.ps1 `
  -VisitType primary `
  -Python .\.venv\Scripts\python.exe `
  -Models GPT-5,o3-mini
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
# The demo ships with and reuses local Hugging Face caches.  Without these
# flags, SentenceTransformer sends HEAD requests even when the model is ready.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"

$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
if (-not $CaseRoot) { $CaseRoot = Join-Path $RepoRoot "data_500" }
if (-not $Config) { $Config = Join-Path $RepoRoot "llm_config.json" }
if (-not $Python) {
    $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot ("outputs\diagnosis_demo_{0}" -f $VisitType) }

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
    [System.IO.File]::WriteAllText(
        $Path,
        $json,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Test-EligibleCase {
    param([string]$CaseId)
    $CasePath = Join-Path $CaseRoot $CaseId
    $Required = @("primary_consultation.json", "diagnosis.json")
    if ($VisitType -eq "followup") { $Required += "follow_up_consultation.json" }
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
    if ($null -ne $ConfigEntry.model) {
        $ConfiguredNames += [string]$ConfigEntry.model
    }
    if ($null -ne $ConfigEntry.tags) {
        $ConfiguredNames += @($ConfigEntry.tags | ForEach-Object { [string]$_ })
    }
}
$ConfiguredNames = @($ConfiguredNames | Select-Object -Unique)
$MissingModels = @($ModelTags | Where-Object { $_ -notin $ConfiguredNames })
if ($MissingModels.Count -gt 0) {
    throw "Models absent from config: $($MissingModels -join ', ')"
}

# Validate local model availability before making any paid LLM calls.
Invoke-Python @(
    "-c",
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO', device='cpu'); print('Local feature semantic model ready.')"
)
if (-not $NoRag) {
    Invoke-Python @(
        "-c",
        "from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu'); CrossEncoder('ncbi/MedCPT-Cross-Encoder'); print('Local OrphaCode RAG models ready.')"
    )
}

$SplitRoot = Join-Path $OutputRoot "splits"
$LlmRoot = Join-Path $OutputRoot "llm_outputs"
if (-not $ScoreRoot) { $ScoreRoot = Join-Path $OutputRoot "judge_scores" }
$FeatureRoot = Join-Path $OutputRoot "features"
$ModelRoot = Join-Path $OutputRoot "model"
$ResultRoot = Join-Path $OutputRoot "results"
@($SplitRoot, $LlmRoot, $FeatureRoot, $ModelRoot, $ResultRoot) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

if (-not $TrainIds) {
    $TrainIds = Join-Path $SplitRoot "train.json"
    $TestIds = Join-Path $SplitRoot "test.json"
    # Keep the -c program on one line and use only Python single-quoted strings.
    # Windows PowerShell 5.1 can otherwise split embedded double-quoted source
    # fragments into extra native-process arguments.
    $AutoSplitCode = "import json,random,sys; from pathlib import Path; root=Path(sys.argv[1]); stage=sys.argv[2]; n_cases=int(sys.argv[3]); train_count=int(sys.argv[4]); seed=int(sys.argv[5]); out_dir=Path(sys.argv[6]); required=['primary_consultation.json','diagnosis.json']+(['follow_up_consultation.json'] if stage=='followup' else []); ids=sorted(p.name for p in root.iterdir() if p.is_dir() and all((p/name).is_file() for name in required)); assert 5<=train_count<n_cases<=len(ids),f'require 5 <= train-count < num-cases <= {len(ids)}'; random.Random(seed).shuffle(ids); chosen=ids[:n_cases]; out_dir.mkdir(parents=True,exist_ok=True); [(out_dir/name).write_text(json.dumps(values,indent=2),encoding='utf-8') for name,values in [('all.json',chosen),('train.json',chosen[:train_count]),('test.json',chosen[train_count:])]]"
    Invoke-Python @(
        "-c", $AutoSplitCode, $CaseRoot, $VisitType, $NumCases, $TrainCount, $Seed, $SplitRoot
    )
    $AllCaseIds = Read-CaseIds (Join-Path $SplitRoot "all.json")
    $TrainCaseIds = Read-CaseIds $TrainIds
    $TestCaseIds = Read-CaseIds $TestIds
} else {
    $TrainCaseIds = Read-CaseIds $TrainIds
    $TestCaseIds = Read-CaseIds $TestIds
    $Overlap = @($TrainCaseIds | Where-Object { $_ -in $TestCaseIds })
    if ($TrainCaseIds.Count -lt 5) { throw "At least five training cases are required." }
    if ($TestCaseIds.Count -eq 0) { throw "Test split must not be empty." }
    if ($Overlap.Count -gt 0) { throw "Train/test overlap: $($Overlap -join ', ')" }
    $AllCaseIds = @($TrainCaseIds + $TestCaseIds)
}

$TrainIds = Join-Path $SplitRoot "train.json"
$TestIds = Join-Path $SplitRoot "test.json"
$AllIds = Join-Path $SplitRoot "all.json"
Write-CaseIds $TrainIds $TrainCaseIds
Write-CaseIds $TestIds $TestCaseIds
Write-CaseIds $AllIds $AllCaseIds

$InvalidIds = @($AllCaseIds | Where-Object { -not (Test-EligibleCase $_) })
if ($InvalidIds.Count -gt 0) {
    throw "Ineligible or missing cases: $($InvalidIds -join ', ')"
}

Write-Host "RareDiagnosis code demo"
Write-Host "  stage:       $VisitType"
Write-Host "  cases:       $AllIds"
Write-Host "  train/test:  $TrainIds / $TestIds"
Write-Host "  models:      $($ModelTags -join ',')"
Write-Host "  output:      $OutputRoot"

$RagArgs = @()
if (-not $NoRag) {
    $RagCache = Join-Path (Split-Path -Parent $ScriptDir) "orphacode_rag_cache"
    if (-not (Test-Path -LiteralPath $RagCache -PathType Container)) {
        throw "OrphaCode RAG cache does not exist: $RagCache"
    }
    $RagArgs = @(
        "--enable-orphacode-rag",
        "--rag-ontology-path", (Join-Path $ScriptDir "orphanet_hierarchy.json"),
        "--rag-vector-cache-dir", $RagCache
    )
}

foreach ($ModelTag in $ModelTags) {
    if ($VisitType -eq "followup") {
        Invoke-Python (@(
            "-m", "rare_diagnosis.training.generate_llm_outputs", $CaseRoot, $LlmRoot,
            "--model", $ModelTag, "--config", $Config, "--case-ids", $AllIds,
            "--visit-type", "primary", "--num-workers", $NumWorkers
        ) + $RagArgs)
    }

    Invoke-Python (@(
        "-m", "rare_diagnosis.training.generate_llm_outputs", $CaseRoot, $LlmRoot,
        "--model", $ModelTag, "--config", $Config, "--case-ids", $AllIds,
        "--visit-type", $VisitType, "--num-workers", $NumWorkers
    ) + $RagArgs)

}

# The LLM-as-judge implementation is intentionally not distributed.  Reproduce
# it from the paper, then write the resulting JSON files under $ScoreRoot before
# rerunning this script.  Candidate generation is resumable, so existing LLM
# outputs are not requested again on the next run.
$ScoreFile = if ($VisitType -eq "primary") { "primary_diagnosis_score.json" } else { "followup_diagnosis_score.json" }
$MissingScores = @()
foreach ($ModelTag in $ModelTags) {
    foreach ($CaseId in $AllCaseIds) {
        $ExpectedScore = Join-Path (Join-Path (Join-Path $ScoreRoot $ModelTag) $CaseId) $ScoreFile
        if (-not (Test-Path -LiteralPath $ExpectedScore -PathType Leaf)) {
            $MissingScores += $ExpectedScore
        }
    }
}
if ($MissingScores.Count -gt 0) {
    $Example = $MissingScores[0]
    throw ("LLM-as-judge code is not included in this repository. Reproduce the judge described in the paper and provide score JSON files before feature construction. Expected: {0}. Missing {1} file(s)." -f $Example, $MissingScores.Count)
}

if ($VisitType -eq "primary") {
    $FeatureModule = "rare_diagnosis.training.build_features_primary"
    $BestConfig = Join-Path $ScriptDir "best_hyperopt_config_primary.json"
    $ExtraFeatureArgs = @()
} else {
    $FeatureModule = "rare_diagnosis.training.build_features_followup"
    $BestConfig = Join-Path $ScriptDir "best_hyperopt_config_followup.json"
    $ExtraFeatureArgs = @("--followup_fname", "followup_diagnosis_orphacode.json")
}

Invoke-Python (@(
    "-m", $FeatureModule,
    "--query_root", $CaseRoot, "--primary_models_root", $LlmRoot,
    "--score_root", $ScoreRoot, "--train_ids", $TrainIds,
    "--test_ids", $TestIds, "--out_dir", $FeatureRoot,
    "--models", $ModelsCsv,
    "--ontology_path", (Join-Path $ScriptDir "orphanet_hierarchy.json"),
    "--num_gpus", $FeatureGpus, "--workers", $FeatureWorkers
) + $ExtraFeatureArgs)

$TrainArgs = @(
    "-m", "rare_diagnosis.training.train_ranker",
    "--input-dir", $FeatureRoot, "--config", $BestConfig,
    "--out-dir", $ModelRoot, "--results-dir", $ResultRoot
)
if ($UseGpu) { $TrainArgs += "--use-gpu" }
Invoke-Python $TrainArgs

Write-Host "Demo complete: $OutputRoot"

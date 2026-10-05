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
    [string]$ScoreRoot = "",
    [string]$Python = "",
    [int]$NumWorkers = 4,
    [int]$FeatureGpus = 0,
    [switch]$UseGpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
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
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "outputs\treatment_demo" }

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
    return (Test-Path -LiteralPath (Join-Path $CasePath "treatment_plan.json") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CasePath "treatment_outcome.json") -PathType Leaf)
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
$ConfiguredNames = @($ConfiguredNames | Select-Object -Unique)
$MissingModels = @($ModelTags | Where-Object { $_ -notin $ConfiguredNames })
if ($MissingModels.Count -gt 0) {
    throw "Models absent from config: $($MissingModels -join ', ')"
}

# Validate local feature models before making paid LLM calls.
Invoke-Python @(
    "-c",
    "from huggingface_hub import snapshot_download; from sentence_transformers import CrossEncoder,SentenceTransformer; s=snapshot_download('pritamdeka/S-PubMedBert-MS-MARCO',local_files_only=True); n=snapshot_download('cross-encoder/nli-deberta-v3-large',local_files_only=True); SentenceTransformer(s,device='cpu'); CrossEncoder(n,device='cpu'); print('Local Treatment feature models ready.')"
)

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
    $AutoSplitCode = "import json,random,sys; from pathlib import Path; root=Path(sys.argv[1]); n=int(sys.argv[2]); nt=int(sys.argv[3]); seed=int(sys.argv[4]); out=Path(sys.argv[5]); ids=sorted(p.name for p in root.iterdir() if p.is_dir() and (p/'treatment_plan.json').is_file() and (p/'treatment_outcome.json').is_file()); assert 5<=nt<n<=len(ids),f'require 5 <= train-count < num-cases <= {len(ids)}'; random.Random(seed).shuffle(ids); chosen=ids[:n]; out.mkdir(parents=True,exist_ok=True); [(out/name).write_text(json.dumps(values,indent=2),encoding='utf-8') for name,values in [('all.json',chosen),('train.json',chosen[:nt]),('test.json',chosen[nt:])]]"
    Invoke-Python @("-c", $AutoSplitCode, $CaseRoot, $NumCases, $TrainCount, $Seed, $SplitRoot)
    $AllCaseIds = Read-CaseIds (Join-Path $SplitRoot "all.json")
    $TrainCaseIds = Read-CaseIds $TrainIds
    $TestCaseIds = Read-CaseIds $TestIds
} else {
    $TrainCaseIds = Read-CaseIds $TrainIds
    $TestCaseIds = Read-CaseIds $TestIds
    $Overlap = @($TrainCaseIds | Where-Object { $_ -in $TestCaseIds })
    if ($TrainCaseIds.Count -lt 5) { throw "At least five training cases are required." }
    if ($TestCaseIds.Count -eq 0) { throw "Test split must not be empty." }
    if (@($TrainCaseIds | Select-Object -Unique).Count -ne $TrainCaseIds.Count) {
        throw "Train split contains duplicate case IDs."
    }
    if (@($TestCaseIds | Select-Object -Unique).Count -ne $TestCaseIds.Count) {
        throw "Test split contains duplicate case IDs."
    }
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
    throw "Ineligible or missing treatment cases: $($InvalidIds -join ', ')"
}

Write-Host "RareTreatment code demo"
Write-Host "  cases:       $AllIds"
Write-Host "  train/test:  $TrainIds / $TestIds"
Write-Host "  models:      $($ModelTags -join ',')"
Write-Host "  output:      $OutputRoot"

foreach ($ModelTag in $ModelTags) {
    Invoke-Python @(
        "-m", "rare_treatment.training.generate_llm_outputs", $CaseRoot, $LlmRoot,
        "--model", $ModelTag, "--config", $Config, "--case-ids", $AllIds,
        "--num-workers", $NumWorkers
    )

}

# The LLM-as-judge implementation is intentionally not distributed. Reproduce
# it from the paper, then write the resulting JSON files under $ScoreRoot before
# rerunning this script. Candidate generation is resumable, so existing LLM
# outputs are not requested again on the next run.
$MissingScores = @()
foreach ($ModelTag in $ModelTags) {
    foreach ($CaseId in $AllCaseIds) {
        $ExpectedScore = Join-Path (Join-Path (Join-Path $ScoreRoot $ModelTag) $CaseId) "treatment_score.json"
        if (-not (Test-Path -LiteralPath $ExpectedScore -PathType Leaf)) {
            $MissingScores += $ExpectedScore
        }
    }
}
if ($MissingScores.Count -gt 0) {
    $Example = $MissingScores[0]
    throw ("LLM-as-judge code is not included in this repository. Reproduce the judge described in the paper and provide score JSON files before feature construction. Expected: {0}. Missing {1} file(s)." -f $Example, $MissingScores.Count)
}

Invoke-Python @(
    "-m", "rare_treatment.training.build_features",
    "--case_root", $CaseRoot, "--llm_root", $LlmRoot, "--score_root", $ScoreRoot,
    "--train_ids", $TrainIds, "--test_ids", $TestIds,
    "--out_dir", $FeatureRoot, "--num_gpus", $FeatureGpus
)

$TrainArgs = @(
    "-m", "rare_treatment.training.train_ranker",
    "--data-dir", $FeatureRoot, "--out-dir", $ModelRoot,
    "--results-dir", $ResultRoot, "--objective", "rank:ndcg",
    "--n-splits", 5, "--target-k", 3, "--save-models"
)
if (-not $UseGpu) { $TrainArgs += "--force-cpu" }
Invoke-Python $TrainArgs

Write-Host "Demo complete: $OutputRoot"

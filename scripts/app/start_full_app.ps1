param(
    [string]$BindHost = "127.0.0.1",
    [int]$BindPort = 8000,
    [switch]$Reload,
    [ValidateSet("yolov8", "yolo_world")]
    [string]$DetectorBackend = "yolov8",
    [string]$ModelWeights = "",
    [switch]$SkipOllama,
    [switch]$SkipVectorCheck,
    [switch]$AutoPullModel
)

$argsList = @(
    "scripts/app/start_full_app.py",
    "--host", $BindHost,
    "--port", $BindPort,
    "--detector-backend", $DetectorBackend
)

if ($Reload) {
    $argsList += "--reload"
}

if ($ModelWeights) {
    $argsList += @("--model-weights", $ModelWeights)
}

if ($SkipOllama) {
    $argsList += "--skip-ollama"
}

if ($SkipVectorCheck) {
    $argsList += "--skip-vector-check"
}

if ($AutoPullModel) {
    $argsList += "--auto-pull-model"
}

python @argsList
exit $LASTEXITCODE

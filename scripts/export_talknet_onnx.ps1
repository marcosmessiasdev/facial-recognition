param(
  [string]$PythonDir = ".tmp_python_embed_3.11.9",
  [string]$TalkNetRepoDir = ".tmp_talknet/TalkNet-ASD-main",
  [string]$WeightsPath = "App/models/talknet/pretrain_TalkSet.model",
  [string]$OutOnnxPath = "App/onnx/talknet_asd.onnx"
)

$py = Join-Path $PythonDir "python.exe"
if(-not (Test-Path $py)){ throw "Python not found at: $py" }
if(-not (Test-Path $TalkNetRepoDir)){ throw "TalkNet repo dir not found at: $TalkNetRepoDir" }
if(-not (Test-Path $WeightsPath)){ throw "Weights not found at: $WeightsPath" }

& $py "scripts/export_talknet_onnx.py" `
  --talknet_repo $TalkNetRepoDir `
  --weights $WeightsPath `
  --out $OutOnnxPath

Write-Host "Exported: $OutOnnxPath"


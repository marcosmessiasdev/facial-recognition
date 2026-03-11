param(
    [switch]$ValidateOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = Split-Path -Parent $PSScriptRoot
    return $root
}

function Ensure-Dir([string]$path) {
    $dir = Split-Path -Parent $path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
}

function Get-Sha256([string]$path) {
    return (Get-FileHash -Algorithm SHA256 -Path $path).Hash.ToLowerInvariant()
}

function Download-File([string]$url, [string]$dest) {
    Ensure-Dir $dest
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        Write-Host "Downloading $url"
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing | Out-Null
        Move-Item -Force $tmp $dest
    } finally {
        if (Test-Path $tmp) { Remove-Item -Force $tmp -ErrorAction SilentlyContinue }
    }
}

$root = Get-RepoRoot
Push-Location $root

try {
    $items = @(
        @{
            Path   = "App/onnx/scrfd_2.5g_kps.onnx"
            Sha256 = "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9"
            Url    = $null # vendored in repo; keep offline
        },
        @{
            Path   = "App/onnx/arcface.onnx"
            Sha256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
            Url    = "https://dl.photoprism.app/onnx/models/w600k_r50.onnx"
        },
        @{
            Path   = "App/onnx/emotion-ferplus-8.onnx"
            Sha256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
            Url    = $null # mirror varies; vendored
        },
        @{
            Path   = "App/onnx/genderage.onnx"
            Sha256 = "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"
            Url    = $null # vendored
        },
        @{
            Path   = "App/onnx/silero_vad.onnx"
            Sha256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
            Url    = "https://raw.githubusercontent.com/snakers4/silero-vad/master/files/silero_vad.onnx"
        },
        @{
            Path   = "App/onnx/face_mesh_Nx3x192x192_post.onnx"
            Sha256 = "ae6ada9004f22be3ab6bab8cbfbc8b795f8595f69e5eea77b95cb2fc344c04eb"
            Url    = $null # vendored
        },
        @{
            Path   = "App/onnx/nemo_en_titanet_small.onnx"
            Sha256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"
            Url    = $null # vendored
        },
        @{
            Path   = "App/onnx/talknet_asd.onnx"
            Sha256 = "7a80b18497359db20aa651fcaba168d3aaeb67f4c6d09575cd8b149813161f40"
            Url    = $null # exported; vendored
        },
        @{
            Path   = "App/models/whisper/ggml-tiny.bin"
            Sha256 = $null # large file; vendored (validated by existence)
            Url    = $null
        }
    )

    $ok = 0
    $bad = 0

    foreach ($it in $items) {
        $path = $it.Path
        $sha = $it.Sha256
        $url = $it.Url

        if (-not (Test-Path $path)) {
            if ($ValidateOnly -or -not $url) {
                Write-Host "MISSING: $path"
                $bad++
                continue
            }

            Download-File $url $path
        }

        if ($sha) {
            $actual = Get-Sha256 $path
            if ($actual -ne $sha) {
                Write-Host "HASH MISMATCH: $path"
                Write-Host "  expected: $sha"
                Write-Host "  actual:   $actual"
                if (-not $ValidateOnly -and $url -and $Force) {
                    Write-Host "Re-downloading due to -Force..."
                    Download-File $url $path
                    $actual2 = Get-Sha256 $path
                    if ($actual2 -ne $sha) {
                        $bad++
                        continue
                    }
                    $ok++
                    continue
                }
                $bad++
                continue
            }
        }

        Write-Host "OK: $path"
        $ok++
    }

    if ($bad -gt 0) {
        throw "Model validation failed: $bad item(s) missing or mismatched; $ok OK."
    }

    Write-Host "All models validated: $ok OK."
}
finally {
    Pop-Location
}


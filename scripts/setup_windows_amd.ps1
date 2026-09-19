# Build the native-Windows AMD ROCm training environment for gfx1100 (RX 7900 XTX).
#
# The AMD "TheRock" index below is a self-contained PEP 503 index that serves
# Windows ROCm builds of torch plus the rocm-sdk runtime DLLs for gfx110X.
$ErrorActionPreference = 'Stop'

$root = 'C:\Users\heyiy\Desktop\cangying'
$venv = Join-Path $root '.venv'
$amdIndex = 'https://d2awnip2yjpvqn.cloudfront.net/v2/gfx110X-dgpu/'

if (-not (Test-Path $venv)) {
    Write-Host '=== creating venv ==='
    python -m venv $venv
}
$py = Join-Path $venv 'Scripts\python.exe'

Write-Host '=== upgrading pip ==='
& $py -m pip install --upgrade pip

Write-Host '=== installing ROCm torch for gfx110X (this is the big download) ==='
& $py -m pip install torch torchvision torchaudio --index-url $amdIndex

Write-Host '=== installing the rest from PyPI ==='
& $py -m pip install numpy pandas pyarrow requests httpx tqdm einops tokenizers `
    fastapi "uvicorn[standard]" jinja2

Write-Host '=== pinned versions ==='
& $py -c "import torch, sys; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('hip', getattr(torch.version, 'hip', None)); print('cuda_api_available', torch.cuda.is_available())"

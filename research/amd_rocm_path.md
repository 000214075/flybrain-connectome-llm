# PyTorch training on RX 7900 XTX (gfx1100) on Windows 11 — findings and recommendation

Researched 2026-09-16. One line per finding, each with a source URL. Items I could not verify are flagged **UNCONFIRMED**.

---

## 1. ROCm on WSL for gfx1100 — official status, driver, install, and the /dev/dri question

- **Yes, AMD officially supports ROCm on WSL for the RX 7900 XTX (gfx1100)** — it is listed by name in the ROCDXG WSL compatibility matrix for librocdxg 1.2.0 / ROCm 7.2.x — https://github.com/ROCm/librocdxg
- The same repo's verification step prints literally `Name: gfx1100  Marketing Name: Radeon RX 7900 XTX`, so gfx1100 is the reference device for WSL bring-up, not an afterthought — https://github.com/ROCm/librocdxg
- Supported WSL host/distro set: **Windows 11** with **Ubuntu 26.04 / 24.04 / 22.04 LTS** — https://github.com/ROCm/librocdxg
- **Exact Windows driver required for the current stack: AMD Software Adrenalin Edition 26.2.2** (released 2026-02-26); its release notes state "Support for WSL with ROCm 7.2.1 now uses the open-source ROCDXG library" and list RX 7900 series as compatible — https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html
- **Exact ROCm version pairing: Adrenalin 26.2.2 + ROCm 7.2.1** is the AMD-stated production combination (librocdxg 1.2.0 also lists ROCm 7.2.x generally) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- **Critical mechanism change:** WSL ROCm no longer needs Radeon Software for Linux packages or `amdgpu-dkms`; ROCDXG is a *user-mode* library that talks to Microsoft's DXCore interface `/dev/dxg`, and "the Windows display driver remains the authoritative GPU driver" — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- **Therefore `/dev/dri` and `/dev/kfd` are NOT expected to exist in WSL, and their absence is normal** — WSL does GPU semi-virtualization through `/dev/dxg`; `/dev/dri` and `lspci` output are absent by design — https://github.com/microsoft/WSL/issues/40464
- Corroborated: the standard `hsa-rocr` package "uses libdrm and talks to `/dev/kfd` directly — this doesn't work in WSL2"; the WSL-specific runtime `hsa-runtime-rocr4wsl-amdgpu` has DXG support built in — https://github.com/andweng/wsl-rocm
- Group membership (`render`/`video`) and `--device=/dev/kfd --device=/dev/dri` are the **native-Linux bare-metal** requirements (ROCm install prerequisites / Docker examples), not WSL requirements — https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
- **Required env var for legacy ROCm releases (< 7.13): `HSA_ENABLE_DXG_DETECTION=1`** — it makes the HSA runtime detect the GPU via `/dev/dxg` and load `librocdxg`; the requirement is removed starting with TheRock 7.13 — https://github.com/ROCm/librocdxg
- **Legacy (ROCm 7.2) in-WSL install procedure:** `amdgpu-install -y --usecase=wsl,rocm --no-dkms` (note `--no-dkms`; no in-WSL kernel module is installed) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-radeon.html
- **New (ROCm 7.2.1+) procedure:** install the Windows driver → install WSL → build/install librocdxg from https://github.com/ROCm/librocdxg → install frameworks using the Radeon Linux how-tos — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- librocdxg install options: **Option A** build from source (needs Windows SDK + CMake ≥ 3.15 + GCC ≥ 11.4), **Option B** prebuilt `rocdxg-roct_<version>_amd64.deb` from the librocdxg GitHub Releases — https://github.com/ROCm/librocdxg
- **Known WSL failure modes when `rocminfo` shows no GPU** (these — not missing `/dev/dri` — are the real causes):
  - Missing/old Windows Adrenalin driver without WSL support → no `/dev/dxg`; `/dev/dxg` should exist inside WSL (user's machine already has it) — https://github.com/ROCm/librocdxg
  - librocdxg not installed / not loading (`/opt/rocm/lib/librocdxg.so`) — https://github.com/ROCm/librocdxg
  - `HSA_ENABLE_DXG_DETECTION=1` not set on ROCm < 7.13 — https://github.com/ROCm/librocdxg
  - Ubuntu 24.04's ancient ROCm 5.7 system libs (`libhsa-runtime64-1`, `libhsakmt1`) shadow `/opt/rocm/*/lib` and always fail with "**ROCk module is NOT loaded**" → `sudo apt remove libhsa-runtime64-1 libhsakmt1` and ensure `hsa-rocr` (standard runtime) is not installed — https://github.com/andweng/wsl-rocm
  - **PyTorch wheels ship their own `libhsa-runtime64.so` which must be deleted** for WSL: `rm libhsa-runtime64.so*` inside `<site-packages>/torch/lib/` — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html
- **Directly relevant to the local machine:** ROCm 7.2.4 was installed via the generic ROCm Linux quick-start, which does not add the WSL pieces — the gaps are the WSL HSA runtime (`hsa-runtime-rocr4wsl-amdgpu`, from the `repo.radeon.com/amdgpu/...` repo), librocdxg, and `HSA_ENABLE_DXG_DETECTION=1` — https://github.com/andweng/wsl-rocm
- Also missing on the local machine per the same source: `PATH`/`LD_LIBRARY_PATH` are not pointing at `/opt/rocm*/bin` and `/opt/rocm*/lib`, which is why `rocminfo`/`rocm-smi`/`hipcc` are not found — https://github.com/andweng/wsl-rocm
- `amdgpu-dkms` has no role in WSL; it is a native-Linux package, and the WSL usecase is explicitly installed with `--no-dkms` — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-radeon.html

## 2. Official PyTorch ROCm wheels — is gfx1100 supported?

- **Yes — gfx1100 is in the official supported-GPU list for PyTorch ROCm wheels: `gfx1030, gfx1100, gfx1101, gfx1102, gfx1200, gfx1201, gfx900, gfx906, gfx908, gfx90a, gfx942`** — https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html
- **Because gfx1100 is officially supported, `HSA_OVERRIDE_GFX_VERSION` is NOT needed for gfx1100 with official PyTorch ROCm wheels — do not set it** — https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html
- The value sometimes cited in community guides, `HSA_OVERRIDE_GFX_VERSION=11.0.0`, is the gfx1100 native version and is a no-op for gfx1100; it is a workaround for *unsupported* GCN/RDNA parts, not for this card — https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU
- Stable index `https://download.pytorch.org/whl/rocm6.4` is real and documented by AMD; PyTorch 2.9 adds WheelNext "variant wheels" for **ROCm 6.3/6.4 only** — https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html
- PyTorch nightly non-variant wheels are what carry newer ROCm on pytorch.org: `pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0` (and a `rocm7.2` nightly index likewise) — https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html
- **AMD explicitly recommends repo.radeon.com wheels over pytorch.org for Radeon:** "The ROCm WHLs available at PyTorch.org are not tested extensively by AMD as the WHLs change regularly" — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html
- **Most reliable gfx1100 combination found: PyTorch 2.9.1 + ROCm 7.2.x** — matching Radeon wheels `torch-2.9.1+rocm7.2.0`, `torchvision-0.24.0`, `torchaudio-2.9.0`, and `triton-3.5.1+rocm7.2.0` (cp312, linux_x86_64) from `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/` — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html
- AMD's own validated ROCm 7.2.4 image for that release: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1` — https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
- Caveat for Python 3.12 on the Radeon wheels: `numpy` 2.0 is incompatible, pin `numpy==1.26.4` — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html
- Whether the *stable* (non-variant) `rocm7.x` pytorch.org indexes also list gfx1100 explicitly is **UNCONFIRMED**; the documented gfx1100-in-wheel-list statement is tied to the ROCm 6.3/6.4 variant wheels — https://www.amd.com/en/developer/resources/technical-articles/2025/pytorch-2-9-wheel-variant-support-expands-to-rocm.html

## 3. ROCm/HIP on native Windows with PyTorch — does it exist?

- **Yes, official PyTorch-on-Windows-with-ROCm wheels exist** for Python 3.12: `torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl`, `torchaudio-2.9.1+rocm7.2.1`, `torchvision-0.24.1+rocm7.2.1`, plus `rocm_sdk_core/devel/libraries_custom-7.2.1` and `rocm-7.2.1.tar.gz`, all from `repo.radeon.com/rocm/windows/rocm-rel-7.2.1/` — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
- Prerequisite: **Adrenalin 26.2.2 graphics driver** must be installed for the 7.2.1 Windows PyTorch release — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
- AMD's own example verification output for Windows uses `device name [0]: Radeon RX 7900 XTX`, so gfx1100 is a supported Windows target — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html
- **Decisive blocker: AMD documents "No ML training support" on Windows** for ROCm PyTorch, along with "only LLM batch sizes of 1 are officially supported" and "On Windows, only Pytorch is supported, not the entire ROCm stack" — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html
- Also Windows-only: Python 3.12 only; may need Visual C++ 2015-2022 redistributables; WDAG and Smart App Control must be disabled — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html
- PyTorch is the only framework AMD lists for Windows Radeon; Linux/WSL gets PyTorch + TensorFlow + JAX + ONNX — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html
- Community self-contained Windows PyTorch wheels for gfx1100 exist (TheRock builds by @jammm, Python 3.12) but are explicitly "not tested on all hardware" — https://github.com/scottt/rocm-TheRock/releases

## 4. Throughput comparison for small-transformer training (≈10M–100M params, bf16/fp16)

- **No published, credible head-to-head benchmark of *training* on gfx1100 across WSL-ROCm / Windows-ROCm / DirectML / Vulkan was found — treat all relative-throughput numbers below as indicative, and the training comparison as UNCONFIRMED.**
- The only concrete gfx1100 ROCm-vs-Vulkan measurement found is *inference* (llama.cpp, Qwen 27B Q4_K_M, 32k context, Ubuntu): **ROCm 7.2.4 prefill 713.4 tok/s vs Vulkan+Mesa/RADV 25.3.5 635.5 tok/s; ROCm decode 44.0 tok/s vs Vulkan 57.9–60.3 tok/s** — i.e. same order of magnitude, ROCm ahead on prefill, behind on decode — https://github.com/hcgdw/rx-7900-xtx-llm-benchmark/blob/main/README.zh-CN.md
- Same source: ROCm 7.13 preview did **not** beat ROCm 7.2.4, and enabling **rocWMMA was catastrophically slower** (ROCm 7.2.4 prefill 713.4 → 254.6 tok/s, decode 44.0 → 16.6 tok/s) — avoid rocWMMA on this card — https://github.com/hcgdw/rx-7900-xtx-llm-benchmark/blob/main/README.zh-CN.md
- ROCm on WSL uses the same ROCm userspace as native Linux (only the kernel-mode path differs, via `/dev/dxg`), so WSL throughput should track native-Linux ROCm closely; the DXG translation layer's exact overhead for training is **UNCONFIRMED** — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html

Feature matrix (as far as I could confirm):

| Option | bf16/fp16 AMP | SDPA / flash attn | torch.compile | Fused optimizers |
|---|---|---|---|---|
| (a) ROCm in WSL2 | Yes (bf16, fp16, fp8 in ROCm 6.4 data-type list) — https://rocm.docs.amd.com/en/docs-6.4.1/compatibility/ml-compatibility/pytorch-compatibility.html | Via AOTriton, but **disabled by default on RX 7000**; must set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html | Yes — Triton ships as a ROCm wheel (`triton-3.5.1+rocm7.2.0`) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html | Yes (`apex` 1.9.0+rocm7.2.4 in AMD's validated image) — https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html |
| (b) ROCm native Windows | Wheels exist (bf16 path same PyTorch build) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html | **UNCONFIRMED** (AOTriton/Triton not shipped for Windows) | No — no Triton on Windows; **UNCONFIRMED** but no Windows Triton wheel is documented | **UNCONFIRMED** |
| (c) torch-directml (Windows) | Limited (no fp8/bf16 parity; operator coverage gaps) — https://pypi.org/project/torch-directml/ | No | No | No |
| (d) Vulkan | No PyTorch training backend (llama.cpp-class inference only) — https://github.com/hcgdw/rx-7900-xtx-llm-benchmark/blob/main/README.zh-CN.md | n/a | No | No |

- **DirectML is in maintenance mode** — "no new functionality or feature updates are planned"; Microsoft now points Windows 11 24H2+ users to Windows ML — https://github.com/microsoft/DirectML
- torch-directml's newest artifacts are `0.2.5.dev240914` uploaded **2024-09-15** — effectively ~2 years stale as of 2026 and pinned to an old PyTorch — https://pypi.org/project/torch-directml/
- Net effect for training: options (c) and (d) are not viable for 10M–100M-param transformer training in bf16/fp16; only (a) and (b) are real, and (b) is officially inference-only — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html

## 5. Known gfx1100-specific blockers

- **Flash attention / SDPA:** "AO Triton with PyTorch 2.9 is disabled by default for AMD Radeon RX 7000 series graphics products, and must be enabled manually" — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html
- The enable flag is `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; AMD-described as ~19x attention speedup on a related RDNA3.5 part — https://github.com/andweng/wsl-rocm
- Upstream `flash-attention` (Dao-AILab) ships no gfx1100/ROCm build; the practical route on RDNA3 is a Triton reimplementation — https://github.com/chelokot/flash-attention-rdna3
- AMD's own Radeon feature note says **FlashAttention-2 backward pass** is enabled on Linux, i.e. it is a *training*-relevant feature that is on Linux/WSL, not Windows — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html
- **rocWMMA:** measured as a large regression on 7900 XTX in the only gfx1100 data point found — leave it off — https://github.com/hcgdw/rx-7900-xtx-llm-benchmark/blob/main/README.zh-CN.md
- **Triton:** available on Linux/WSL as `triton-3.5.1+rocm7.2.0` (and `pytorch-triton-rocm==3.5.0` via variant wheels); **no Triton on native Windows**, so `torch.compile`/Inductor is a WSL/Linux-only feature — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html
- **bitsandbytes:** a ROCm/AMD backend is documented by Hugging Face for recent versions, and ROCm forks exist; **whether the current release's AMD backend covers gfx1100 end-to-end is UNCONFIRMED** — https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=AMD+ROCm and https://github.com/bitsandbytes-foundation/bitsandbytes
- **xformers:** **UNCONFIRMED** — no official ROCm/gfx1100 build was found.
- **`hipMallocManaged` unsupported under WSL** (reported on RDNA3.5): ROCm can only allocate dedicated VRAM, and WSL has pinned memory disabled by default, which also causes `RuntimeError: UVA is not available` in some loaders — https://github.com/andweng/wsl-rocm
- **Profiling/monitoring unavailable under WSL:** `amd-smi` is limited; `rocprofiler` and the ROCm debugger are not supported — https://github.com/ROCm/librocdxg
- **JAX is not validated under WSL** and may fail to install or execute — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- **MIGraphX and mGPU configuration are not supported under WSL** — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html

---

## Ranking (most to least recommended)

1. **ROCm 7.2.x inside WSL2 via ROCDXG/librocdxg** — the only officially *training*-capable, gfx1100-listed, Triton/`torch.compile`-capable option on this machine.
2. **ROCm in WSL2 via AMD's prebuilt Docker image** (`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1`) — same stack, fewer host-side moving parts; AMD's recommended method for reliability.
3. **Native Linux install of ROCm 7.2.x** (dual boot) — most robust and fully supported, but not WSL/Windows.
4. **ROCm PyTorch on native Windows** — works and is official for gfx1100 *inference only*; AMD states "No ML training support". Use only if WSL is impossible and you accept unsupported training.
5. **torch-directml** — maintenance-mode stack, ~2-year-old artifacts, weak operator/bf16 coverage. Not recommended for training.
6. **Vulkan** — no PyTorch training backend at all.

## Single best path

**WSL2 + ROCm 7.2.x + ROCDXG/librocdxg + PyTorch 2.9.1+rocm7.2 Radeon wheels.**

### Step 0 — Windows host
Install **AMD Software: Adrenalin Edition 26.2.2** (required; provides DXCore `/dev/dxg`): https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html
Then, in PowerShell, confirm WSL is current:
```powershell
wsl --update
wsl --version
```

### Step 1 — Inside WSL (Ubuntu 24.04), install the WSL-specific HSA runtime
```bash
sudo apt update
sudo apt install -y cmake gcc g++ git

# Ubuntu 24.04 ships ancient ROCm 5.7 libs that shadow /opt/rocm and cause "ROCk module is NOT loaded"
sudo apt remove -y libhsa-runtime64-1 libhsakmt1
sudo apt remove -y hsa-rocr            # conflicts with the WSL runtime

# WSL-specific HSA runtime with DXG support built in
sudo apt install -y hsa-runtime-rocr4wsl-amdgpu
dpkg -l | grep rocr4wsl                # expect: hsa-runtime-rocr4wsl-amdgpu
```
(Source: https://github.com/andweng/wsl-rocm and https://github.com/ROCm/librocdxg)

### Step 2 — Install librocdxg (the DXG bridge library)
```bash
export HSA_ENABLE_DXG_DETECTION=1

# Option B (fastest): prebuilt deb from https://github.com/ROCm/librocdxg/releases
sudo dpkg -i rocdxg-roct_<version>_amd64.deb

# Option A (from source, requires Windows SDK installed on the Windows side)
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg
export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0'
mkdir -p build && cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make -j"$(nproc)"
sudo make install
```
(Source: https://github.com/ROCm/librocdxg)

### Step 3 — Environment variables (append to `~/.bashrc`)
```bash
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:$LD_LIBRARY_PATH
export PATH=/opt/rocm-7.2.4/bin:/opt/rocm/bin:$PATH
```
(Source: https://github.com/andweng/wsl-rocm — adjust the version to your `/opt/rocm-7.2.4`)

### Step 4 — Verify the GPU is visible
```bash
rocminfo | grep -E "Name:|Marketing Name:"
```
Expected: `gfx1100` / `Radeon RX 7900 XTX`, preceded by `Load librocdxg.so successully!` and `WSL environment detected.`
(Source: https://github.com/ROCm/librocdxg)

### Step 5 — Install PyTorch 2.9.1 + ROCm 7.2 for gfx1100
```bash
sudo apt install -y python3-pip python3-dev libjpeg-dev
pip3 install --upgrade pip wheel
pip3 install numpy==1.26.4

wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/torchaudio-2.9.0%2Brocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl

pip3 install --break-system-packages torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl \
    torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
    torchaudio-2.9.0+rocm7.2.0.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
    triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl

# REQUIRED under WSL: drop the bundled HSA runtime so the WSL one is used
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
rm -f "${location}/torch/lib/"libhsa-runtime64.so*
```
(Source: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html)

### Step 6 — Training-time env and verification
```bash
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # RX 7000 has AOTriton disabled by default on PyTorch 2.9

python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True Radeon RX 7900 XTX
```
Do **not** set `HSA_OVERRIDE_GFX_VERSION` — gfx1100 is officially supported. Avoid rocWMMA. Watch for WSL's disabled pinned memory / missing UVA if a loader errors out.
(Sources: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsrad.html, https://github.com/andweng/wsl-rocm)

### Lower-effort alternative to Steps 1–5
Run AMD's validated image with the WSL device/library passthrough:
```bash
sudo apt install -y docker.io
sudo docker run -it \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 8G \
  --device=/dev/dxg \
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
  -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
  -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1
```
(Sources: https://github.com/ROCm/librocdxg and https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)

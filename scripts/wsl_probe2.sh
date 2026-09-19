#!/usr/bin/env bash
# Probe the WSL-side ROCm <-> dxg bridge, which is what ROCm on WSL actually uses.
echo "=== librocdxg present? ==="
find / -name 'librocdxg*' 2>/dev/null | head
echo "=== ROCr libs ==="
ls -l /opt/rocm/lib/libhsa-runtime64.so* /opt/rocm/lib/libamdhip64.so* 2>&1 | head
echo "=== rocm lib dir sample ==="
ls /opt/rocm/lib 2>&1 | head -40
echo "=== rocm dir top-level ==="
ls /opt/rocm-7.2.4 2>&1
echo "=== is this the WSL build of rocm? ==="
ls /opt/rocm/.info 2>&1
cat /opt/rocm/.info/version 2>&1
cat /opt/rocm/.info/version-dev 2>&1
echo "=== dpkg rocm packages ==="
dpkg -l 2>/dev/null | grep -iE 'rocm|hip|hsa|dxg' | awk '{print $2, $3}' | head -40
echo "=== apt sources mentioning radeon/rocm ==="
cat /etc/apt/sources.list.d/*.list 2>/dev/null | head -20
cat /etc/apt/sources.list.d/*.sources 2>/dev/null | head -30
echo "=== try rocminfo with HSA_ENABLE_DXG_DETECTION=1 ==="
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo 2>&1 | head -25
echo "=== try with HSA_OVERRIDE_GFX_VERSION=11.0.0 ==="
HSA_OVERRIDE_GFX_VERSION=11.0.0 HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo 2>&1 | head -15
echo "=== d3d12 / dxcore libs ==="
ls -l /usr/lib/wsl/lib/ 2>&1 | head -20
echo "=== LD_LIBRARY_PATH ==="
echo "$LD_LIBRARY_PATH"

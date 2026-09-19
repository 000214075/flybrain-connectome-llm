#!/usr/bin/env bash
# Targeted probe of the WSL-side ROCm <-> dxg bridge. No full-filesystem walks.
echo "=== librocdxg in rocm dirs ==="
ls -l /opt/rocm/lib/librocdxg* 2>&1
ls -l /opt/rocm/lib/libhsa-runtime64.so* 2>&1
ls -l /opt/rocm/lib/libamdhip64.so* 2>&1
echo "=== rocm lib listing (first 50) ==="
ls /opt/rocm/lib 2>&1 | head -50
echo "=== rocm version info ==="
cat /opt/rocm/.info/version 2>&1
cat /opt/rocm/.info/version-dev 2>&1
echo "=== dpkg rocm/hip packages ==="
dpkg -l 2>/dev/null | grep -iE 'rocm|hip|hsa|dxg|amdgpu' | awk '{print $2, $3}' | head -40
echo "=== apt sources ==="
ls /etc/apt/sources.list.d/ 2>&1
echo "=== wsl libs ==="
ls -l /usr/lib/wsl/lib/ 2>&1 | head -20
echo "=== rocminfo plain ==="
timeout 60 /opt/rocm/bin/rocminfo 2>&1 | head -20
echo "=== rocminfo with DXG detect ==="
HSA_ENABLE_DXG_DETECTION=1 timeout 60 /opt/rocm/bin/rocminfo 2>&1 | head -20
echo "=== rocm_agent_enumerator ==="
timeout 60 /opt/rocm/bin/rocm_agent_enumerator 2>&1 | head -10
echo "=== env ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PATH=$PATH"

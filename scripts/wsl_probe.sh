#!/usr/bin/env bash
# Probe GPU/ROCm availability inside WSL.
echo "=== uname ==="
uname -a
echo "=== GPU device nodes ==="
ls -l /dev/dri 2>&1
ls -l /dev/kfd 2>&1
ls -l /dev/dxg 2>&1
echo "=== rocm tree ==="
ls -d /opt/rocm* 2>&1
echo "=== rocm bin ==="
ls /opt/rocm/bin 2>&1 | head -40
echo "=== amdgpu-arch ==="
/opt/rocm/bin/amdgpu-arch 2>&1 | head -5
echo "=== rocminfo (head) ==="
/opt/rocm/bin/rocminfo 2>&1 | head -40
echo "=== kernel modules ==="
ls /lib/modules 2>&1
echo "=== modules matching amd/dxg ==="
find /lib/modules -maxdepth 2 -name '*amdgpu*' 2>/dev/null | head
find /lib/modules -maxdepth 2 -name '*dxg*' 2>/dev/null | head
echo "=== dmesg-ish: /sys/class/drm ==="
ls /sys/class/drm 2>&1
echo "=== python ==="
python3 --version
python3 -m pip list 2>/dev/null | grep -iE 'torch|numpy|transformers|gradio|fastapi' || echo "(no relevant pkgs)"
echo "=== groups ==="
id
echo "=== disk ==="
df -h /
echo "=== where is the ext4 vhdx (host side) ==="
cat /proc/mounts | grep -E 'drvfs|9p|ext4' | head -10

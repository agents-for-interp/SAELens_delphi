#!/bin/bash
# Script to release RAM and GPU memory on the remote instance

echo "=== Memory Status Before ==="
free -h
echo ""
echo "=== GPU Memory Status ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "No GPU detected"
echo ""

echo "=== Clearing GPU Memory ==="
# Kill any Python processes that might be holding GPU memory
pkill -f "python.*torch" 2>/dev/null || echo "No Python torch processes found"
sleep 1

# Reset GPU using nvidia-smi (if available)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU memory will be released when processes are killed"
    nvidia-smi --gpu-reset 2>/dev/null || echo "GPU reset requires root or is not available"
fi

echo ""
echo "=== Clearing System Caches ==="
# Clear page cache, dentries, and inodes (requires root)
if [ "$EUID" -eq 0 ]; then
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && echo "✓ System caches cleared" || echo "Could not clear system caches"
else
    echo "Note: System cache clearing requires root privileges"
    echo "Run: sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
fi

echo ""
echo "=== Memory Status After ==="
free -h
echo ""
echo "=== GPU Memory Status After ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "No GPU detected"


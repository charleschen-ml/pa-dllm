#!/bin/bash
# Script to try allocating GPUs from best to worst
# Usage: ./get_best_gpu.sh

set -e

# Configuration
PARTITION="coc-gpu,ice-gpu"
QOS="coc-ice"     # Changed from coc-grade to coc-ice for separate quota
NUM_GPUS=1        # Number of GPUs to request (1, 2, 4, etc.)
CPUS=2
MEM="128G"
TIME="16:00:00"
WAIT_TIME=30  # seconds to wait for each GPU type

# GPU types in order of preference (best to worst)
# Note: NUM_GPUS will be appended to each type
GPU_TYPES=(
    "h200"        # NVIDIA H200 - newest, most powerful
    "h100"        # NVIDIA H100 - excellent
    "a100"        # NVIDIA A100 - very good
    "l40s"        # NVIDIA L40S - great for inference
    "mi210"       # AMD MI210 - decent
    "a40"         # NVIDIA A40 - older
    "rtx_6000"    # NVIDIA RTX 6000 - workstation
    "v100"        # NVIDIA V100 - oldest but usable
)

GPU_NAMES=(
    "H200"
    "H100"
    "A100"
    "L40S"
    "MI210"
    "A40"
    "RTX 6000"
    "V100"
)

echo "========================================="
echo "GPU Allocation Script"
echo "========================================="
echo "Configuration:"
echo "  Partition: $PARTITION"
echo "  QOS: $QOS"
echo "  GPUs: $NUM_GPUS"
echo "  CPUs: $CPUS"
echo "  Memory: $MEM"
echo "  Time: $TIME"
echo "  Wait per GPU: ${WAIT_TIME}s"
echo "========================================="
echo ""

# Function to try allocating a GPU
try_allocate() {
    local gpu_type=$1
    local gpu_name=$2
    local gpu_spec="${gpu_type}:${NUM_GPUS}"
    
    echo "🔍 Trying ${NUM_GPUS}x ${gpu_name}..."
    echo "   Command: salloc -p $PARTITION --qos=$QOS --gres=gpu:${gpu_spec} --ntasks=1 --cpus-per-task=$CPUS --mem=$MEM -t $TIME"
    
    # Get count of current jobs before submission
    JOBS_BEFORE=$(squeue -u $USER -h | wc -l)
    
    # Start allocation in background
    salloc -p $PARTITION --qos=$QOS --gres=gpu:${gpu_spec} --ntasks=1 --cpus-per-task=$CPUS --mem=$MEM -t $TIME &
    SALLOC_PID=$!
    
    # Wait for job to appear in queue
    sleep 2
    
    # Get the newest job (highest job ID) that appeared after submission
    JOBID=$(squeue -u $USER -h -t PENDING,RUNNING -o "%i" -S "-i" | head -1)
    
    if [ -z "$JOBID" ]; then
        echo "   ❌ Failed to submit job"
        kill $SALLOC_PID 2>/dev/null || true
        return 1
    fi
    
    echo "   ⏳ Job $JOBID submitted, waiting ${WAIT_TIME}s..."
    
    # Wait and check status
    for ((i=1; i<=$WAIT_TIME; i++)); do
        sleep 1
        
        # Check if job is running
        JOB_STATE=$(squeue -j $JOBID -h -o "%T" 2>/dev/null || echo "NOTFOUND")
        
        if [ "$JOB_STATE" == "RUNNING" ]; then
            echo "   ✅ SUCCESS! Allocated ${NUM_GPUS}x ${gpu_name} (Job $JOBID)"
            
            # Cancel the background salloc job - we'll reconnect properly
            echo "   🔄 Reconnecting to allocation..."
            scancel $JOBID 2>/dev/null || true
            kill $SALLOC_PID 2>/dev/null || true
            wait $SALLOC_PID 2>/dev/null || true
            sleep 1
            
            echo ""
            echo "========================================="
            echo "🎉 GPU ALLOCATED!"
            echo "========================================="
            echo "GPUs: ${NUM_GPUS}x ${gpu_name}"
            echo "Partition: $PARTITION"
            echo ""
            echo "Starting interactive session on GPU node..."
            echo "You can now run commands directly (e.g., nvidia-smi)"
            echo "Type 'exit' when done to release the allocation."
            echo "========================================="
            echo ""
            
            # Run salloc in foreground - this gives you a direct shell on the GPU node
            # No need for srun - you'll be directly on the compute node
            exec salloc -p $PARTITION --qos=$QOS --gres=gpu:${gpu_spec} --ntasks=1 --cpus-per-task=$CPUS --mem=$MEM -t $TIME
            
            # This line won't be reached due to exec
            exit 0
        elif [ "$JOB_STATE" == "NOTFOUND" ]; then
            echo "   ❌ Job disappeared (may have failed)"
            kill $SALLOC_PID 2>/dev/null || true
            return 1
        fi
        
        # Show progress every 10 seconds
        if [ $((i % 10)) -eq 0 ]; then
            REASON=$(squeue -j $JOBID -h -o "%r" 2>/dev/null || echo "Unknown")
            echo "   ⏳ Still waiting... (${i}s/${WAIT_TIME}s, Reason: $REASON)"
        fi
    done
    
    # Timeout - cancel and try next
    echo "   ⏰ Timeout after ${WAIT_TIME}s"
    echo "   🚫 Canceling job $JOBID..."
    scancel $JOBID 2>/dev/null || true
    kill $SALLOC_PID 2>/dev/null || true
    sleep 2  # Give it time to cancel
    
    return 1
}

# Try each GPU type
for i in "${!GPU_TYPES[@]}"; do
    gpu_type="${GPU_TYPES[$i]}"
    gpu_name="${GPU_NAMES[$i]}"
    
    if try_allocate "$gpu_type" "$gpu_name"; then
        exit 0
    fi
    
    echo ""
done

# If we get here, nothing worked
echo "========================================="
echo "❌ FAILED TO ALLOCATE ANY GPU"
echo "========================================="
echo "All GPU types exhausted. The cluster may be fully utilized."
echo ""
echo "Suggestions:"
echo "  1. Try again later (peak hours: 9am-5pm)"
echo "  2. Request shorter time (reduces wait)"
echo "  3. Check cluster status: sinfo -p $PARTITION"
echo "  4. Check queue: squeue -p $PARTITION"
echo "========================================="
exit 1


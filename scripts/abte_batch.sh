#!/bin/bash

# Define the base command
BASE_CMD="python src/train_ABTE.py --batch 64 --epochs 10 --lr 3e-5"

# Define an array of CUDA device IDs
DEVICES=(0 1 2 3)

# Define the combinations of --lr_schedule and --adapter
declare -a LR_SCHEDULE_OPTS=("" "--lr_schedule")
declare -a ADAPTER_OPTS=("" "--adapter")

# Initialize device index
DEVICE_IDX=0

echo "Starting training runs with different configurations..."

# Loop through lr_schedule options
for lr_sched_opt in "${LR_SCHEDULE_OPTS[@]}"; do
    # Loop through adapter options
    for adapter_opt in "${ADAPTER_OPTS[@]}"; do
        # Construct the full command for the current combination
        CURRENT_CMD="$BASE_CMD $lr_sched_opt $adapter_opt"

        # Set CUDA_VISIBLE_DEVICES for the current run
        export CUDA_VISIBLE_DEVICES=${DEVICES[DEVICE_IDX]}

        echo "-----------------------------------------------------"
        echo "Executing on CUDA device ${DEVICES[DEVICE_IDX]}:"
        echo "$CURRENT_CMD"
        echo "-----------------------------------------------------"

        # Execute the command
        $CURRENT_CMD &

        # Increment device index
        DEVICE_IDX=$((DEVICE_IDX + 1))

        # Optional: Add a small delay to ensure processes start cleanly,
        # or if you want them to run sequentially, remove the '&' and this sleep.
        # sleep 5
    done
done

echo "All training processes have been launched."
echo "You can use 'nvidia-smi' to monitor GPU usage."
echo "To wait for all background processes to finish, you can use 'wait' command after this script exits."

# If you want the script to wait for all background processes to finish before exiting, uncomment the line below:
# wait
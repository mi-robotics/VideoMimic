#!/bin/bash

# Simple script to process TRAM data and convert to PHC format
TRAM_DATA_DIR="/home/mcarroll/Documents/cd-2/VideoMimic/src/videomimic/data/video_mimic_demos/tram"
OUTPUT_DIR="/home/mcarroll/Documents/cd-2/VideoMimic/src/videomimic/data/video_mimic_demos/phc"
PYTHON_SCRIPT="/home/mcarroll/Documents/cd-2/VideoMimic/src/videomimic/data_process/tram_to_phc_format.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Processing TRAM data from: $TRAM_DATA_DIR"
echo "Output directory: $OUTPUT_DIR"


python3 "$PYTHON_SCRIPT" --path "$TRAM_DATA_DIR" --output "$OUTPUT_DIR/tram_data.pkl" --upright_start
echo "Conversion complete!"

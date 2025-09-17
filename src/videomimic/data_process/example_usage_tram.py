#!/usr/bin/env python3
"""
Example usage of the TRAM video processing script

This script demonstrates how to use the TRAM video processor
"""

import subprocess
import sys
from pathlib import Path

def run_tram_processing_example():
    """
    Example of how to run TRAM processing on a folder of videos
    """
    
    # Path to the TRAM processor script
    processor_script = Path(__file__).parent / "process_videos_tram.py"
    
    # Example: Process videos from a folder
    input_folder = "/path/to/your/mp4/videos"  # Change this to your video folder
    output_folder = "/home/milo/Documents/phd/VideoMimic/src/videomimic/data/demo_video_ds/tram"
    
    # Command to run the processor
    cmd = [
        sys.executable, str(processor_script),
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--static_camera",  # Uncomment if cameras are static
        "--max_humans", "10"  # Adjust based on your needs
    ]
    
    print("Example command:")
    print(" ".join(cmd))
    print("\nTo run this command:")
    print("1. Update the input_folder path to point to your MP4 files")
    print("2. Run the command above")
    print("\nOr run directly:")
    print(f"python {processor_script} --input_folder /path/to/videos --output_folder {output_folder}")

if __name__ == "__main__":
    run_tram_processing_example()

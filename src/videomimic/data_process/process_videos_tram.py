#!/usr/bin/env python3
"""
TRAM Video Processing Script

This script processes a folder of MP4 files using the TRAM mocap pipeline:
1. Ensures videos are 30fps
2. Runs camera estimation
3. Runs human pose estimation  
4. Organizes outputs in the specified directory

Usage:
    python process_videos_tram.py --input_folder /path/to/mp4/folder --output_folder /src/videomimic/data/demo_video_ds/tram
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import tempfile
import logging

# Add TRAM to path
TRAM_ROOT = Path(__file__).parent.parent.parent.parent / "tram"
sys.path.insert(0, str(TRAM_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def ensure_30fps(input_video, output_video):
    """
    Ensure video is 30fps using ffmpeg
    
    Args:
        input_video (str): Path to input video
        output_video (str): Path to output video
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', '-i', input_video,
            '-r', '30',  # Set frame rate to 30fps
            '-c:v', 'libx264',  # Use H.264 codec
            '-preset', 'fast',  # Fast encoding
            '-y',  # Overwrite output file
            output_video
        ]
        
        logger.info(f"Converting {input_video} to 30fps...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted to 30fps: {output_video}")
            return True
        else:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error converting video to 30fps: {e}")
        return False


def run_tram_camera_estimation(video_path, static_camera=False):
    """
    Run TRAM camera estimation script
    
    Args:
        video_path (str): Path to video file
        static_camera (bool): Whether camera is static
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            'python', str(TRAM_ROOT / 'scripts' / 'estimate_camera.py'),
            '--video', video_path
        ]
        
        if static_camera:
            cmd.append('--static_camera')
            
        logger.info(f"Running camera estimation for {video_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TRAM_ROOT))
        
        if result.returncode == 0:
            logger.info(f"Camera estimation completed successfully")
            return True
        else:
            logger.error(f"Camera estimation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running camera estimation: {e}")
        return False


def run_tram_human_estimation(video_path, max_humans=20):
    """
    Run TRAM human pose estimation script
    
    Args:
        video_path (str): Path to video file
        max_humans (int): Maximum number of humans to reconstruct
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cmd = [
            'python', str(TRAM_ROOT / 'scripts' / 'estimate_humans.py'),
            '--video', video_path,
            '--max_humans', str(max_humans)
        ]
        
        logger.info(f"Running human pose estimation for {video_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TRAM_ROOT))
        
        if result.returncode == 0:
            logger.info(f"Human pose estimation completed successfully")
            return True
        else:
            logger.error(f"Human pose estimation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error running human pose estimation: {e}")
        return False




def organize_tram_outputs(video_name, tram_results_dir, output_dir):
    """
    Organize TRAM outputs into the specified output directory
    
    Args:
        video_name (str): Name of the video (without extension)
        tram_results_dir (str): TRAM results directory
        output_dir (str): Target output directory
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create video-specific output directory
        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy TRAM results
        tram_seq_folder = Path(tram_results_dir) / video_name
        
        if tram_seq_folder.exists():
            # Copy all files from TRAM results
            for item in tram_seq_folder.iterdir():
                if item.is_file():
                    shutil.copy2(item, video_output_dir)
                elif item.is_dir():
                    shutil.copytree(item, video_output_dir / item.name, dirs_exist_ok=True)
            
            logger.info(f"Organized outputs for {video_name} to {video_output_dir}")
            return True
        else:
            logger.error(f"TRAM results folder not found: {tram_seq_folder}")
            return False
            
    except Exception as e:
        logger.error(f"Error organizing outputs: {e}")
        return False


def process_single_video(video_path, output_dir, static_camera=False, max_humans=20, temp_dir=None):
    """
    Process a single video through the TRAM mocap pipeline (camera + human estimation)
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Output directory
        static_camera (bool): Whether camera is static
        max_humans (int): Maximum number of humans
        temp_dir (str): Temporary directory for processing
    
    Returns:
        bool: True if successful, False otherwise
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    
    logger.info(f"Processing video: {video_name}")
    
    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    temp_dir = Path(temp_dir)
    
    try:
        # Step 1: Ensure 30fps
        temp_video = temp_dir / f"{video_name}_30fps.mp4"
        if not ensure_30fps(str(video_path), str(temp_video)):
            logger.error(f"Failed to convert {video_name} to 30fps")
            return False
        
        # Step 2: Run camera estimation
        if not run_tram_camera_estimation(str(temp_video), static_camera):
            logger.error(f"Camera estimation failed for {video_name}")
            return False
        
        # Step 3: Run human pose estimation
        if not run_tram_human_estimation(str(temp_video), max_humans):
            logger.error(f"Human pose estimation failed for {video_name}")
            return False
        
        # Step 4: Organize outputs
        if not organize_tram_outputs(video_name, str(TRAM_ROOT / 'results'), output_dir):
            logger.error(f"Failed to organize outputs for {video_name}")
            return False
        
        logger.info(f"Successfully processed {video_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {video_name}: {e}")
        return False
    
    finally:
        # Clean up temporary video
        if temp_video.exists():
            temp_video.unlink()


def process_video_folder(input_folder, output_folder, static_camera=False, max_humans=20):
    """
    Process all MP4 files in a folder using TRAM mocap pipeline
    
    Args:
        input_folder (str): Path to folder containing MP4 files
        output_folder (str): Path to output folder
        static_camera (bool): Whether cameras are static
        max_humans (int): Maximum number of humans per video
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all MP4 files
    mp4_files = list(input_folder.glob("*.mp4"))
    
    if not mp4_files:
        logger.warning(f"No MP4 files found in {input_folder}")
        return
    
    logger.info(f"Found {len(mp4_files)} MP4 files to process")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        successful = 0
        failed = 0
        
        for video_file in mp4_files:
            logger.info(f"Processing {video_file.name} ({successful + failed + 1}/{len(mp4_files)})")
            
            if process_single_video(
                str(video_file), 
                str(output_folder),
                static_camera=static_camera,
                max_humans=max_humans,
                temp_dir=temp_dir
            ):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Process MP4 videos using TRAM mocap pipeline")
    parser.add_argument("--input_folder", type=str, required=True,
                       help="Path to folder containing MP4 files")
    parser.add_argument("--output_folder", type=str, 
                       default="/home/milo/Documents/phd/VideoMimic/src/videomimic/data/demo_video_ds/tram",
                       help="Path to output folder")
    parser.add_argument("--static_camera", action="store_true",
                       help="Assume cameras are static")
    parser.add_argument("--max_humans", type=int, default=20,
                       help="Maximum number of humans to reconstruct per video")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not Path(args.input_folder).exists():
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return
    
    # Check if TRAM is available
    if not TRAM_ROOT.exists():
        logger.error(f"TRAM directory not found: {TRAM_ROOT}")
        return
    
    logger.info(f"Processing videos from: {args.input_folder}")
    logger.info(f"Output directory: {args.output_folder}")
    logger.info(f"Static camera: {args.static_camera}")
    logger.info(f"Max humans per video: {args.max_humans}")
    
    # Process all videos
    process_video_folder(
        args.input_folder,
        args.output_folder,
        static_camera=args.static_camera,
        max_humans=args.max_humans
    )


if __name__ == "__main__":
    main()

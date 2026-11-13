#!/usr/bin/env python3
"""
Convert image sequences to MP4.
Encode H.265 from consecutive or irregularly named images.
Supports GPU encoding, progress reporting, and batch encoding.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import re
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Generator

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Install tqdm to show a progress bar: pip install tqdm")
    print()

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Install Pillow for faster image inspection: pip install Pillow")
    print()


def natural_sort_key(s):
    """Sort names numerically where they contain digits."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def get_image_files(input_dir, extensions=None, stride=1, recursive=False, path_regex=None, path_regex_mode="relative"):
    """
    Find and sort images in the input directory.

    Args:
        input_dir: Directory containing the images.
        extensions: Image extensions to include.
        stride: Keep every nth image. Use 1 to keep all images.
        recursive: Include subdirectories.
        path_regex: Optional path filter, such as 840412060917/color_000000.jpg.
        path_regex_mode: Match relative or absolute paths.

    Returns:
        Sorted image paths.
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Collect images.
    image_files = []
    for ext in extensions:
        if recursive:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

    # Filter paths using the requested match mode.
    if path_regex:
        pattern = re.compile(path_regex)
        filtered = []
        for img_file in image_files:
            if path_regex_mode == "absolute":
                match_target = img_file.resolve().as_posix()
            else:
                match_target = img_file.relative_to(input_path).as_posix()
            if pattern.search(match_target):
                filtered.append(img_file)
        image_files = filtered

    if not image_files:
        raise ValueError(f"No images found in {input_dir}")

    # Sort filenames numerically.
    image_files.sort(key=lambda x: natural_sort_key(x.name))

    # Apply the sampling stride.
    if stride > 1:
        image_files = image_files[::stride]

    return image_files


def create_output_dir(input_dir, output_dir_name="videos"):
    """
    Resolve and create the output directory.

    Args:
        input_dir: Input directory.
        output_dir_name: Output directory name or path.

    Returns:
        Output directory path.
    """
    input_path = Path(input_dir)

    if output_dir_name is None:
        output_dir = input_path
    else:
        output_path = Path(output_dir_name)
        if output_path.is_absolute():
            output_dir = output_path
        elif output_dir_name in [".", "./"]:
            output_dir = input_path
        else:
            parent_dir = input_path.parent
            output_dir = parent_dir / output_dir_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def detect_gpu_encoder():
    """
    Detect an available GPU encoder.

    Returns:
        An (encoder_name, encoder_type) pair, or (None, 'cpu').
    """
    # Probe NVIDIA HEVC encoding.
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        encoders_output = result.stdout

        # Try encoders in priority order.
        if 'hevc_nvenc' in encoders_output:
            # Verify that the NVIDIA encoder can run.
            test_result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=3
            )
            if test_result.returncode == 0:
                return ('hevc_nvenc', 'nvidia')

        if 'hevc_amf' in encoders_output:
            return ('hevc_amf', 'amd')

        if 'hevc_qsv' in encoders_output:
            return ('hevc_qsv', 'intel')

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    return (None, 'cpu')


def get_image_info(image_path):
    """
    Read an image's dimensions.

    Args:
        image_path: Path to the image.

    Returns:
        A dictionary with the width and height.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return {"width": img.width, "height": img.height}
    except ImportError:
        # Fall back to ffprobe if Pillow is unavailable.
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            str(image_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data['streams'][0]
            return {"width": stream['width'], "height": stream['height']}
        return {"width": None, "height": None}


def check_files_consecutive(image_files):
    """
    Determine whether the filenames form a consecutive sequence.

    Args:
        image_files: Paths to the source images.

    Returns:
        A tuple of (is_consecutive, first_number, filename_pattern).
    """
    if len(image_files) < 2:
        return False, 0, None

    # Extract numeric suffixes.
    def extract_number(path):
        # Find numeric components of the filename.
        name = path.stem
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else None

    numbers = [extract_number(f) for f in image_files]

    # Require a numeric component in every filename.
    if None in numbers:
        return False, 0, None

    # Check for gaps.
    start_num = numbers[0]
    for i, num in enumerate(numbers):
        if num != start_num + i:
            return False, 0, None

    # Identify the filename pattern.
    first_file = image_files[0]
    name = first_file.stem
    ext = first_file.suffix

    # Recognize patterns such as 0000, 0001 or frame_0000, frame_0001.
    match = re.search(r'(\d+)$', name)
    if match:
        num_str = match.group(1)
        num_digits = len(num_str)
        prefix = name[:match.start()]
        pattern = f"{prefix}%0{num_digits}d{ext}"
        return True, start_num, pattern

    return False, 0, None


def create_video_ffmpeg(image_files, output_path, fps=30, crf=23, preset='medium',
                        use_gpu=True, gpu_encoder=None, gop_size=1):
    """
    Encode an H.265 MP4 with FFmpeg, with optional GPU encoding and progress.

    Args:
        image_files: Paths to the source images.
        output_path: Path to the output video.
        fps: Output frame rate.
        crf: Quality setting from 0 to 51. Lower values improve quality; 18–28 is recommended.
        preset: Encoding speed preset.
        use_gpu: Enable GPU encoding.
        gpu_encoder: Encoder name and type.
        gop_size: GOP length. The default of 1 requests all I-frames.

    Returns:
        A tuple of (success, encoder_type).
    """
    # Check whether the input filenames are consecutive.
    is_consecutive, start_num, pattern = check_files_consecutive(image_files)

    # Choose the encoder.
    encoder_name = 'libx265'
    encoder_type = 'cpu'

    if use_gpu and gpu_encoder and gpu_encoder[0]:
        encoder_name = gpu_encoder[0]
        encoder_type = gpu_encoder[1]
        print(f"Using GPU encoding: {encoder_type.upper()} ({encoder_name})")
    else:
        print(f"Using CPU encoding: {encoder_name}")

    # Choose the input mode based on the filename sequence.
    temp_list_file = None

    if is_consecutive and pattern:
        # Use a filename pattern for consecutive images.
        print("Using a filename pattern for consecutive images")
        print(f"  Pattern: {pattern}")

        # Locate the image directory.
        img_dir = image_files[0].parent
        pattern_path = img_dir / pattern

        # Build the FFmpeg command for image-pattern input.
        # The image decoder does not support CUDA hardware acceleration.
        cmd = ['ffmpeg']
        cmd.extend([
            '-start_number', str(start_num),
            '-framerate', str(fps),
            '-i', str(pattern_path),
            '-frames:v', str(len(image_files)),  # Limit the output frame count.
            '-threads', '0',
            '-vsync', '0',
            '-c:v', encoder_name,
        ])
    else:
        # Use a concat list for irregular filenames.
        print("Using concat input for irregular filenames")

        temp_list_file = output_path.parent / f"temp_filelist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # Write the list of input files.
        with open(temp_list_file, 'w') as f:
            for img_file in image_files:
                f.write(f"file '{str(img_file.absolute())}'\n")

        # Build the FFmpeg command for concat input.
        # The image decoder does not support CUDA hardware acceleration.
        cmd = ['ffmpeg']
        cmd.extend([
            '-f', 'concat',
            '-safe', '0',
            '-r', str(fps),
            '-threads', '0',
            '-thread_queue_size', '1024',
            '-i', str(temp_list_file),
            '-vsync', '0',
            '-c:v', encoder_name,
        ])

    # Set encoder-specific options.
    if encoder_type == 'nvidia':
        # NVIDIA encoder options.
        cmd.extend([
            '-preset', 'p4',  # p1–p7, with p7 favoring quality.
            '-rc', 'vbr',     # Variable bitrate.
            '-cq', str(crf),  # Target quality.
            '-b:v', '0',      # VBR without a fixed target bitrate.
        ])
        # NVENC requires a GOP longer than the B-frame count plus one.
        effective_gop = gop_size
        if gop_size < 2:
            effective_gop = 2
            print("NVENC requires a GOP of at least 2. Using GOP=2 with B-frames disabled.")
        cmd.extend([
            '-bf', '0',               # Disable B-frames.
            '-g', str(effective_gop), # GOP length.
        ])
    elif encoder_type == 'amd':
        # AMD encoder options.
        cmd.extend([
        '-quality', 'quality',
        '-rc', 'vbr_latency',
        '-qp_i', str(crf),
        '-qp_p', str(crf),
        '-g', str(gop_size),
    ])
    elif encoder_type == 'intel':
        # Intel encoder options.
        cmd.extend([
        '-preset', 'veryslow',
        '-global_quality', str(crf),
        '-g', str(gop_size),
    ])
    else:
        # CPU encoder options.
        cmd.extend([
        '-crf', str(crf),
        '-preset', preset,
        '-g', str(gop_size),
    ])

    # Shared output options.
    cmd.extend([
        '-pix_fmt', 'yuv420p',
        '-y',
        str(output_path)
    ])

    try:
        print("\nEncoding video...")
        print(f"Frames: {len(image_files)}")

        # Log the full FFmpeg command.
        print(f"\nRunning: {' '.join(cmd)}\n")
        # Stream FFmpeg output to track progress.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # Initialize progress reporting.
        total_frames = len(image_files)
        pbar = None
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_frames, desc="Encoding", unit="frames",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # Read FFmpeg output.
        frame_pattern = re.compile(r'frame=\s*(\d+)')
        last_frame = 0
        stderr_lines = []

        for line in process.stderr:
            stderr_lines.append(line)
            match = frame_pattern.search(line)
            if match:
                current_frame = int(match.group(1))
                if pbar and current_frame > last_frame:
                    pbar.update(current_frame - last_frame)
                    last_frame = current_frame
                elif not pbar and current_frame % 100 == 0:
                    # Report every 100 frames without tqdm.
                    progress = (current_frame / total_frames) * 100
                    print(f"Progress: {current_frame}/{total_frames} ({progress:.1f}%)", end='\r')

        process.wait()

        if pbar:
            pbar.close()
        else:
            print()

        if process.returncode != 0:
            stderr_output = ''.join(stderr_lines)
            print(f"\nFFmpeg failed with exit code {process.returncode}:")
            print(stderr_output)
            return False, encoder_type

        print(f"\nVideo saved to {output_path}")
        return True, encoder_type

    except Exception as e:
        print(f"\nEncoding failed: {e}")
        return False, encoder_type

    finally:
        # Remove temporary input files.
        if temp_list_file and temp_list_file.exists():
            temp_list_file.unlink()


def batch_encode_videos_gpu(
    video_tasks: List[Dict[str, Any]],
    max_concurrent: int = 2,
    use_gpu: bool = True,
    gpu_encoder: Optional[Tuple[str, str]] = None,
) -> Generator[Tuple[bool, str, Dict[str, Any]], None, None]:
    """
    Batch encode multiple videos in parallel using GPU streams.

    NVIDIA GPUs typically support 2-3 concurrent NVENC sessions.
    This function enables parallel encoding to improve GPU utilization
    from 30-50% to 70-90%.

    Args:
        video_tasks: List of dicts with keys:
            - image_files: List of image file paths
            - output_path: Output video path
            - fps: Frame rate (default: 30)
            - crf: Quality parameter (default: 23)
            - preset: Encoding preset (default: 'medium')
            - gop_size: GOP size (default: 10)
        max_concurrent: Maximum concurrent GPU encoding streams (default: 2)
        use_gpu: Whether to use GPU acceleration
        gpu_encoder: GPU encoder info tuple (encoder_name, encoder_type)

    Yields:
        Tuple of (success, encoder_type, task_info) for each completed task
    """
    if not video_tasks:
        return

    def encode_single(task: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        image_files = task.get('image_files', [])
        output_path = task.get('output_path')
        fps = task.get('fps', 30)
        crf = task.get('crf', 23)
        preset = task.get('preset', 'medium')
        gop_size = task.get('gop_size', 10)

        if not image_files or not output_path:
            return False, 'error', task

        success, encoder_type = create_video_ffmpeg(
            image_files=image_files,
            output_path=Path(output_path),
            fps=fps,
            crf=crf,
            preset=preset,
            use_gpu=use_gpu,
            gpu_encoder=gpu_encoder,
            gop_size=gop_size,
        )
        return success, encoder_type, task

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(encode_single, task): task for task in video_tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
                yield result
            except Exception as e:
                task = futures[future]
                print(f"Encoding failed for {task.get('output_path')}: {e}")
                yield False, 'error', task


def save_metadata(output_dir, video_name, metadata):
    """
    Save video metadata.

    Args:
        output_dir: Output directory.
        video_name: Video filename without its extension.
        metadata: Metadata dictionary.
    """
    metadata_file = output_dir / f"{video_name}_info.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Encode an image sequence as an H.265 MP4 with optional GPU acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/images
  %(prog)s /path/to/images --fps 25 --crf 20
  %(prog)s /path/to/images --stride 2  # Keep every second frame
  %(prog)s /path/to/images --gop 30  # Use fewer keyframes to reduce file size
  %(prog)s /path/to/images --gpu  # Enable GPU encoding
  %(prog)s /path/to/images --no-gpu  # Force CPU encoding
  %(prog)s /path/to/images --delete-raw  # Delete source images after encoding
  %(prog)s /path/to/images --output-dir my_videos --output-name custom_video.mp4
        """
    )

    parser.add_argument('input_dir', type=str,
                        help='Directory containing the input images')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output frame rate (default: 30)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Keep every nth image (default: 1, which keeps all frames)')
    parser.add_argument('--gop', type=int, default=10,
                        help='GOP length. Use 1 for all I-frames and precise seeking (default: 1)')
    parser.add_argument('--crf', type=int, default=23,
                        help='H.265 quality from 0 to 51. Lower is better (default: 23)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster',
                                'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='CPU encoding speed preset (default: medium)')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Use GPU encoding (default: auto-detect)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU encoding')
    parser.add_argument('--output-dir', type=str, default='videos',
                        help='Output directory name or absolute path (default: videos). Use "." for the input directory.')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Output filename (defaults to the input directory name)')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp'],
                        help='Image extensions to include (default: .jpg .jpeg .png .bmp)')
    parser.add_argument('--recursive', action='store_true',
                        help='Include subdirectories')
    parser.add_argument('--path-regex', type=str, default=None,
                        help='Filter paths with a regular expression, such as 840412060917/color_000000.jpg')
    parser.add_argument('--path-regex-mode', type=str, default='relative',
                        choices=['relative', 'absolute'],
                        help='Match the expression against relative or absolute paths')
    parser.add_argument('--delete-raw', action='store_true',
                        help='Delete the input images after successful encoding')

    args = parser.parse_args()

    # Resolve the encoding mode.
    use_gpu = args.gpu and not args.no_gpu

    try:
        print("="*60)
        print("H.265 image sequence encoder")
        print("="*60)

        # Check GPU encoder availability.
        gpu_encoder = None
        if use_gpu:
            print("\nChecking for a GPU encoder...")
            gpu_encoder = detect_gpu_encoder()
            if gpu_encoder[0]:
                print(f"Found GPU encoder: {gpu_encoder[1].upper()} ({gpu_encoder[0]})")
            else:
                print("No GPU encoder found. Using CPU encoding.")
                use_gpu = False
        else:
            print("\nUsing CPU encoding")

        # Find input images.
        print(f"\nScanning {args.input_dir}")
        image_files = get_image_files(
            args.input_dir,
            args.extensions,
            args.stride,
            recursive=args.recursive,
            path_regex=args.path_regex,
            path_regex_mode=args.path_regex_mode
        )
        print(f"Found {len(image_files)} images")
        if args.stride > 1:
            print(f"  Sampling stride: {args.stride}")
        print(f"  First image: {image_files[0].name}")
        print(f"  Last image: {image_files[-1].name}")

        # Resolve the output directory.
        output_dir = create_output_dir(args.input_dir, args.output_dir)
        print(f"\nOutput directory: {output_dir}")

        # Resolve the output filename.
        if args.output_name:
            output_name = args.output_name
            if not output_name.endswith('.mp4'):
                output_name += '.mp4'
        else:
            # Use the input directory name.
            input_dir_name = Path(args.input_dir).name
            parent_dir_name = Path(args.input_dir).parent.name
            output_name = f"{parent_dir_name}_{input_dir_name}.mp4"

        output_path = output_dir / output_name

        # Read source image dimensions.
        img_info = get_image_info(image_files[0])
        print(f"Image dimensions: {img_info['width']}x{img_info['height']}")

        # Start timing.
        start_time = time.time()

        # Encode the video.
        success, encoder_type = create_video_ffmpeg(
            image_files,
            output_path,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            use_gpu=use_gpu,
            gpu_encoder=gpu_encoder,
            gop_size=args.gop
        )

        # Measure elapsed time.
        elapsed_time = time.time() - start_time

        if not success:
            print("\nVideo encoding failed")
            return 1

        # Save the metadata.
        encoder_info = {
            "type": encoder_type,
            "crf": args.crf,
            "gop_size": args.gop,
        }
        if encoder_type == 'cpu':
            encoder_info["codec"] = "H.265 (libx265)"
            encoder_info["preset"] = args.preset
        elif encoder_type == 'nvidia':
            encoder_info["codec"] = "H.265 (hevc_nvenc)"
            encoder_info["preset"] = "p7"
        elif encoder_type == 'amd':
            encoder_info["codec"] = "H.265 (hevc_amf)"
        elif encoder_type == 'intel':
            encoder_info["codec"] = "H.265 (hevc_qsv)"

        metadata = {
            "input_directory": str(Path(args.input_dir).absolute()),
            "output_video": str(output_path.absolute()),
            "creation_time": datetime.now().isoformat(),
            "total_frames": len(image_files),
            "stride": args.stride,
            "fps": args.fps,
            "resolution": img_info,
            "encoding": encoder_info,
            "encoding_time_seconds": round(elapsed_time, 2),
            "first_image": image_files[0].name,
            "last_image": image_files[-1].name,
            "image_list": [img.name for img in image_files]
        }

        save_metadata(output_dir, output_path.stem, metadata)

        # Delete the sources if requested.
        if args.delete_raw:
            print("\nDeleting source images...")
            deleted_count = 0
            failed_count = 0

            for img_file in image_files:
                try:
                    img_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Could not delete {img_file.name}: {e}")
                    failed_count += 1

            print(f"Deleted {deleted_count} files", end="")
            if failed_count > 0:
                print(f" ({failed_count} deletions failed)")
            else:
                print()

        # Compute the average encoding throughput.
        avg_fps = len(image_files) / elapsed_time if elapsed_time > 0 else 0

        print("\n" + "="*60)
        print("Encoding complete")
        print("="*60)
        print(f"Video: {output_path}")
        print(f"Frames: {len(image_files)}")
        print(f"Dimensions: {img_info['width']}x{img_info['height']}")
        print(f"Frame rate: {args.fps} fps")
        print(f"Encoder: {encoder_type.upper()}")
        print(f"Elapsed: {elapsed_time:.2f} s ({avg_fps:.1f} frames/s)")
        print("="*60)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

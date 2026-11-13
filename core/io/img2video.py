#!/usr/bin/env python3
"""
通用图片序列转MP4工具
支持H.265编码，自动处理不连续的图片命名
支持GPU加速、进度条可视化和多线程优化
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
    print("提示: 安装tqdm可获得更好的进度显示 (pip install tqdm)")
    print()

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("提示: 安装Pillow可获得更好的性能 (pip install Pillow)")
    print()


def natural_sort_key(s):
    """自然排序key函数，正确处理数字"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]


def get_image_files(input_dir, extensions=None, stride=1, recursive=False, path_regex=None, path_regex_mode="relative"):
    """
    获取目录中的所有图片文件并排序

    Args:
        input_dir: 输入目录路径
        extensions: 支持的图片扩展名列表
        stride: 采样步长，1表示不跳帧，2表示每隔1帧取1帧
        recursive: 是否递归扫描子目录
        path_regex: 对路径进行正则匹配（如 840412060917/color_000000.jpg）
        path_regex_mode: 正则匹配的路径类型 (relative 或 absolute)

    Returns:
        排序后的图片文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 获取所有图片文件
    image_files = []
    for ext in extensions:
        if recursive:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

    # 对相对路径做正则过滤
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
        raise ValueError(f"在目录 {input_dir} 中未找到图片文件")

    # 自然排序
    image_files.sort(key=lambda x: natural_sort_key(x.name))

    # 应用步长采样
    if stride > 1:
        image_files = image_files[::stride]

    return image_files


def create_output_dir(input_dir, output_dir_name="videos"):
    """
    在输入目录的父目录同级创建输出目录

    Args:
        input_dir: 输入目录路径
        output_dir_name: 输出目录名称

    Returns:
        输出目录路径
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
            parent_dir = input_path.parent  # 父目录
            output_dir = parent_dir / output_dir_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def detect_gpu_encoder():
    """
    检测可用的GPU编码器

    Returns:
        tuple: (encoder_name, encoder_type) 或 (None, 'cpu')
    """
    # 检测NVIDIA GPU (hevc_nvenc)
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        encoders_output = result.stdout

        # 按优先级检测
        if 'hevc_nvenc' in encoders_output:
            # 验证NVIDIA GPU是否可用
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
    获取图片信息（分辨率等）

    Args:
        image_path: 图片路径

    Returns:
        包含宽度和高度的字典
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return {"width": img.width, "height": img.height}
    except ImportError:
        # 如果PIL不可用，使用ffprobe
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
    检查图片文件名是否连续

    Args:
        image_files: 图片文件路径列表

    Returns:
        tuple: (是否连续, 起始编号, 文件名模式)
    """
    if len(image_files) < 2:
        return False, 0, None

    # 提取文件名中的数字
    def extract_number(path):
        # 从文件名中提取数字部分
        name = path.stem
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else None

    numbers = [extract_number(f) for f in image_files]

    # 检查是否所有文件都有数字
    if None in numbers:
        return False, 0, None

    # 检查是否连续
    start_num = numbers[0]
    for i, num in enumerate(numbers):
        if num != start_num + i:
            return False, 0, None

    # 检测文件名模式
    first_file = image_files[0]
    name = first_file.stem
    ext = first_file.suffix

    # 尝试匹配模式 (如 0000, 0001, ... 或 frame_0000, frame_0001, ...)
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
    使用ffmpeg创建H.265编码的MP4视频，支持GPU加速和进度显示

    Args:
        image_files: 图片文件路径列表
        output_path: 输出视频路径
        fps: 帧率
        crf: 质量参数 (0-51, 越小质量越好，推荐18-28)
        preset: 编码速度预设
        use_gpu: 是否使用GPU加速
        gpu_encoder: GPU编码器信息 (encoder_name, encoder_type)
        gop_size: GOP大小，1表示全I帧 (默认: 1)

    Returns:
        tuple: (是否成功, 编码器类型)
    """
    # 检查文件是否连续
    is_consecutive, start_num, pattern = check_files_consecutive(image_files)

    # 确定使用的编码器
    encoder_name = 'libx265'
    encoder_type = 'cpu'

    if use_gpu and gpu_encoder and gpu_encoder[0]:
        encoder_name = gpu_encoder[0]
        encoder_type = gpu_encoder[1]
        print(f"✓ 使用GPU加速: {encoder_type.upper()} ({encoder_name})")
    else:
        print(f"使用CPU编码: {encoder_name}")

    # 根据文件是否连续选择不同的输入方式
    temp_list_file = None

    if is_consecutive and pattern:
        # 使用pattern模式（更快）
        print(f"✓ 检测到连续文件名，使用pattern模式加速")
        print(f"  模式: {pattern}")

        # 获取图片目录
        img_dir = image_files[0].parent
        pattern_path = img_dir / pattern

        # 构建ffmpeg命令 - pattern模式
        # 注意: -hwaccel cuda 对图像序列无效，因为图像解码器不支持CUDA加速
        cmd = ['ffmpeg']
        cmd.extend([
            '-start_number', str(start_num),
            '-framerate', str(fps),
            '-i', str(pattern_path),
            '-frames:v', str(len(image_files)),  # 限制帧数
            '-threads', '0',
            '-vsync', '0',
            '-c:v', encoder_name,
        ])
    else:
        # 使用concat模式（兼容性好）
        print(f"使用concat模式（文件名不连续或无法识别模式）")

        temp_list_file = output_path.parent / f"temp_filelist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # 写入文件列表
        with open(temp_list_file, 'w') as f:
            for img_file in image_files:
                f.write(f"file '{str(img_file.absolute())}'\n")

        # 构建ffmpeg命令 - concat模式
        # 注意: -hwaccel cuda 对图像序列无效，因为图像解码器不支持CUDA加速
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

    # 根据编码器类型添加参数
    if encoder_type == 'nvidia':
        # NVIDIA GPU参数
        cmd.extend([
            '-preset', 'p4',  # p1-p7, p7最高质量
            '-rc', 'vbr',     # 可变比特率
            '-cq', str(crf),  # 质量参数
            '-b:v', '0',      # VBR模式
        ])
        # NVENC要求GOP长度 > B帧数 + 1，且此处禁用B帧
        effective_gop = gop_size
        if gop_size < 2:
            effective_gop = 2
            print("⚠️ NVENC不支持GOP=1，已自动调整为GOP=2（保持无B帧）")
        cmd.extend([
            '-bf', '0',               # 禁用B帧
            '-g', str(effective_gop), # GOP大小
        ])
    elif encoder_type == 'amd':
        # AMD GPU参数
        cmd.extend([
        '-quality', 'quality',
        '-rc', 'vbr_latency',
        '-qp_i', str(crf),
        '-qp_p', str(crf),
        '-g', str(gop_size),  # GOP大小
    ])
    elif encoder_type == 'intel':
        # Intel GPU参数
        cmd.extend([
        '-preset', 'veryslow',
        '-global_quality', str(crf),
        '-g', str(gop_size),  # GOP大小
    ])
    else:
        # CPU参数
        cmd.extend([
        '-crf', str(crf),
        '-preset', preset,
        '-g', str(gop_size),  # GOP大小
    ])

    # 通用参数
    cmd.extend([
        '-pix_fmt', 'yuv420p',
        '-y',
        str(output_path)
    ])

    try:
        print(f"\n正在生成视频...")
        print(f"总帧数: {len(image_files)}")

        # 调试：打印完整命令
        print(f"\n执行命令: {' '.join(cmd)}\n")
        # 使用Popen实时获取进度
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        # 进度跟踪
        total_frames = len(image_files)
        pbar = None
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_frames, desc="编码进度", unit="帧",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # 解析ffmpeg输出
        frame_pattern = re.compile(r'frame=\s*(\d+)')
        last_frame = 0
        stderr_lines = []

        for line in process.stderr:
            stderr_lines.append(line)  # 保存所有stderr输出
            match = frame_pattern.search(line)
            if match:
                current_frame = int(match.group(1))
                if pbar and current_frame > last_frame:
                    pbar.update(current_frame - last_frame)
                    last_frame = current_frame
                elif not pbar and current_frame % 100 == 0:
                    # 没有tqdm时，每100帧打印一次
                    progress = (current_frame / total_frames) * 100
                    print(f"进度: {current_frame}/{total_frames} ({progress:.1f}%)", end='\r')

        process.wait()

        if pbar:
            pbar.close()
        else:
            print()  # 换行

        if process.returncode != 0:
            stderr_output = ''.join(stderr_lines)
            print(f"\nFFmpeg错误 (返回码: {process.returncode}):")
            print(stderr_output)
            return False, encoder_type

        print(f"\n✓ 视频生成成功: {output_path}")
        return True, encoder_type

    except Exception as e:
        print(f"\n编码过程出错: {e}")
        return False, encoder_type

    finally:
        # 清理临时文件
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
    保存视频元数据信息

    Args:
        output_dir: 输出目录
        video_name: 视频文件名（不含扩展名）
        metadata: 元数据字典
    """
    metadata_file = output_dir / f"{video_name}_info.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据已保存: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='将图片序列转换为H.265编码的MP4视频 (支持GPU加速)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /path/to/images
  %(prog)s /path/to/images --fps 25 --crf 20
  %(prog)s /path/to/images --stride 2  # 每隔1帧取1帧，加速2倍
  %(prog)s /path/to/images --gop 30  # GOP=30，更小的文件但seek不精确
  %(prog)s /path/to/images --gpu  # 使用GPU加速
  %(prog)s /path/to/images --no-gpu  # 强制使用CPU
  %(prog)s /path/to/images --delete-raw  # 转换后删除源文件
  %(prog)s /path/to/images --output-dir my_videos --output-name custom_video.mp4
        """
    )

    parser.add_argument('input_dir', type=str,
                        help='输入图片目录路径')
    parser.add_argument('--fps', type=int, default=30,
                        help='输出视频帧率 (默认: 30)')
    parser.add_argument('--stride', type=int, default=1,
                        help='图片采样步长，1=不跳帧，2=每隔1帧取1帧 (默认: 1)')
    parser.add_argument('--gop', type=int, default=10,
                        help='GOP大小(关键帧间隔)，1=全I帧，适合精确seek (默认: 1)')
    parser.add_argument('--crf', type=int, default=23,
                        help='H.265质量参数，0-51，越小质量越好 (默认: 23)')
    parser.add_argument('--preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster',
                                'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='CPU编码速度预设 (默认: medium)')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='使用GPU加速 (默认: 自动检测)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='强制使用CPU编码')
    parser.add_argument('--output-dir', type=str, default='videos',
                        help='输出目录名称或绝对路径 (默认: videos，"." 表示输入目录)')
    parser.add_argument('--output-name', type=str, default=None,
                        help='输出视频文件名 (默认: 使用输入目录名)')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp'],
                        help='支持的图片扩展名 (默认: .jpg .jpeg .png .bmp)')
    parser.add_argument('--recursive', action='store_true',
                        help='递归扫描子目录')
    parser.add_argument('--path-regex', type=str, default=None,
                        help='对路径进行正则匹配 (如 840412060917/color_000000.jpg)')
    parser.add_argument('--path-regex-mode', type=str, default='relative',
                        choices=['relative', 'absolute'],
                        help='正则匹配路径类型 (relative 或 absolute)')
    parser.add_argument('--delete-raw', action='store_true',
                        help='视频转换成功后删除源图片文件 (谨慎使用!)')

    args = parser.parse_args()

    # 处理GPU参数
    use_gpu = args.gpu and not args.no_gpu

    try:
        print("="*60)
        print("图片序列转MP4工具 (H.265编码)")
        print("="*60)

        # 检测GPU
        gpu_encoder = None
        if use_gpu:
            print("\n检测GPU编码器...")
            gpu_encoder = detect_gpu_encoder()
            if gpu_encoder[0]:
                print(f"✓ 检测到GPU: {gpu_encoder[1].upper()} ({gpu_encoder[0]})")
            else:
                print("未检测到GPU，将使用CPU编码")
                use_gpu = False
        else:
            print("\n使用CPU编码模式")

        # 获取图片文件
        print(f"\n扫描目录: {args.input_dir}")
        image_files = get_image_files(
            args.input_dir,
            args.extensions,
            args.stride,
            recursive=args.recursive,
            path_regex=args.path_regex,
            path_regex_mode=args.path_regex_mode
        )
        print(f"✓ 找到 {len(image_files)} 张图片")
        if args.stride > 1:
            print(f"  采样步长: {args.stride} (跳帧采样)")
        print(f"  第一张: {image_files[0].name}")
        print(f"  最后一张: {image_files[-1].name}")

        # 创建输出目录
        output_dir = create_output_dir(args.input_dir, args.output_dir)
        print(f"\n输出目录: {output_dir}")

        # 确定输出文件名
        if args.output_name:
            output_name = args.output_name
            if not output_name.endswith('.mp4'):
                output_name += '.mp4'
        else:
            # 使用输入目录名作为输出文件名
            input_dir_name = Path(args.input_dir).name
            parent_dir_name = Path(args.input_dir).parent.name
            output_name = f"{parent_dir_name}_{input_dir_name}.mp4"

        output_path = output_dir / output_name

        # 获取图片信息
        img_info = get_image_info(image_files[0])
        print(f"图片分辨率: {img_info['width']}x{img_info['height']}")

        # 记录开始时间
        start_time = time.time()

        # 创建视频
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

        # 计算耗时
        elapsed_time = time.time() - start_time

        if not success:
            print("\n✗ 视频生成失败")
            return 1

        # 保存元数据
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

        # 删除源文件（如果指定）
        if args.delete_raw:
            print(f"\n正在删除源图片文件...")
            deleted_count = 0
            failed_count = 0

            for img_file in image_files:
                try:
                    img_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  警告: 无法删除 {img_file.name}: {e}")
                    failed_count += 1

            print(f"✓ 已删除 {deleted_count} 个文件", end="")
            if failed_count > 0:
                print(f" ({failed_count} 个文件删除失败)")
            else:
                print()

        # 计算平均帧率
        avg_fps = len(image_files) / elapsed_time if elapsed_time > 0 else 0

        print("\n" + "="*60)
        print("✓ 处理完成!")
        print("="*60)
        print(f"视频文件: {output_path}")
        print(f"总帧数: {len(image_files)}")
        print(f"分辨率: {img_info['width']}x{img_info['height']}")
        print(f"帧率: {args.fps} fps")
        print(f"编码器: {encoder_type.upper()}")
        print(f"耗时: {elapsed_time:.2f}秒 (平均 {avg_fps:.1f} 帧/秒)")
        print("="*60)

        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

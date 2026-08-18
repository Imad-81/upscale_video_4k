import os
import cv2
import torch
import ffmpeg
import argparse
import numpy as np
from tqdm import tqdm

# ===============================
# CONFIG (Default Settings)
# ===============================
DEFAULT_INPUT = "test.mp4"

# Default Target Resolution (1920x1080)
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

# Fit Mode:
#   "pad"     - Preserve original aspect ratio; adds black borders (pillarbox/letterbox) if aspect ratio differs.
#   "stretch" - Stretch/compress video to exact target dimensions.
#   "crop"    - Scale video to fill target dimensions and crop excess edges.
DEFAULT_FIT_MODE = "pad"

UPSCALE_FACTOR_GPU = 4
UPSCALE_FACTOR_CPU = 3


def parse_resolution(res_str: str) -> tuple[int, int]:
    """
    Parses a resolution string such as '1920x1080', '1920*1080', '1920 1080', or presets like '1080p', '4k'.
    """
    if not res_str:
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    res_str = res_str.strip().lower()
    presets = {
        "480p": (854, 480),
        "sd": (854, 480),
        "720p": (1280, 720),
        "hd": (1280, 720),
        "1080p": (1920, 1080),
        "fhd": (1920, 1080),
        "fullhd": (1920, 1080),
        "1440p": (2560, 1440),
        "2k": (2560, 1440),
        "qhd": (2560, 1440),
        "2160p": (3840, 2160),
        "4k": (3840, 2160),
        "uhd": (3840, 2160),
    }

    if res_str in presets:
        return presets[res_str]

    for delim in ["x", "*", ",", " "]:
        if delim in res_str:
            parts = [p.strip() for p in res_str.split(delim) if p.strip()]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return (int(parts[0]), int(parts[1]))

    raise ValueError(
        f"Invalid resolution format: '{res_str}'. Examples: '1920x1080', '1280x720', '1080p', '4k'."
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI Video Upscaler: Upscale video to a set resolution (default: 1920x1080).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale video to default 1920x1080:
  python upscale_video.py my_video.mp4

  # Upscale video to specific resolution:
  python upscale_video.py my_video.mp4 1920x1080
  python upscale_video.py my_video.mp4 1280x720
  python upscale_video.py my_video.mp4 4k

  # Specify custom output path:
  python upscale_video.py my_video.mp4 1920x1080 -o result.mp4

  # Using flags:
  python upscale_video.py -i my_video.mp4 -r 1920x1080 -o result.mp4
        """,
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help=f"Path to input video file (e.g. video.mp4)",
    )
    parser.add_argument(
        "resolution",
        nargs="?",
        default=None,
        help=f"Target resolution (e.g. 1920x1080, 1080p, 4k, 1280x720). Default: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_flag",
        default=None,
        help="Path to input video (alternative to positional argument)",
    )
    parser.add_argument(
        "-r", "--resolution-flag",
        dest="res_flag",
        default=None,
        help=f"Target resolution (e.g. 1920x1080, default: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output video (default: auto-named from input and resolution)",
    )
    parser.add_argument(
        "--width", type=int, default=None, help=f"Target width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=None, help=f"Target height in pixels"
    )
    parser.add_argument(
        "--fit-mode",
        choices=["pad", "stretch", "crop"],
        default=DEFAULT_FIT_MODE,
        help=f"Aspect ratio fit mode: 'pad' (black bars), 'stretch', or 'crop' (default: {DEFAULT_FIT_MODE})",
    )
    return parser.parse_args()


def resize_and_fit(frame, target_w, target_h, fit_mode="pad"):
    """
    Resizes a super-resolved frame to the exact target resolution (target_w, target_h)
    according to the specified fit mode.
    """
    h, w = frame.shape[:2]

    if fit_mode == "stretch":
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    elif fit_mode == "crop":
        scale = max(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        start_x = max(0, (new_w - target_w) // 2)
        start_y = max(0, (new_h - target_h) // 2)
        cropped = resized[start_y : start_y + target_h, start_x : start_x + target_w]

        if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
            cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return cropped

    else:  # "pad" (default - preserves original aspect ratio with pillarbox/letterbox)
        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            final_frame = cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        else:
            final_frame = resized

        if final_frame.shape[0] != target_h or final_frame.shape[1] != target_w:
            final_frame = cv2.resize(final_frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return final_frame


def main():
    args = parse_arguments()

    # 1. Resolve Input Video Path
    input_video = args.video_path or args.input_flag
    if not input_video:
        if os.path.exists(DEFAULT_INPUT):
            input_video = DEFAULT_INPUT
        elif os.isatty(0):
            try:
                user_input = input("Enter video path: ").strip()
                if user_input:
                    input_video = user_input
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Operation cancelled.")
                exit(0)

    if not input_video or not os.path.exists(input_video):
        print(f"[ERROR] Input video not found: '{input_video}'")
        print("\nUsage:")
        print("  python upscale_video.py <video_path> [resolution]")
        print(f"  Example: python upscale_video.py my_video.mp4 1920x1080 (default resolution: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT})")
        exit(1)

    # 2. Resolve Target Resolution (Default: 1920x1080)
    res_input = args.resolution or args.res_flag
    if not res_input and (args.width is not None and args.height is not None):
        target_w, target_h = args.width, args.height
    elif res_input:
        try:
            target_w, target_h = parse_resolution(res_input)
        except ValueError as e:
            print(f"[ERROR] {e}")
            exit(1)
    else:
        target_w, target_h = DEFAULT_WIDTH, DEFAULT_HEIGHT

    # 3. Resolve Output Video Path
    if args.output:
        output_video = args.output
    else:
        base, ext = os.path.splitext(input_video)
        if not ext:
            ext = ".mp4"
        output_video = f"{base}_{target_w}x{target_h}{ext}"

    fit_mode = args.fit_mode

    print("=" * 55)
    print("🎬 AI VIDEO UPSCALER")
    print("=" * 55)
    print(f"[INFO] Input Video:       {input_video}")
    print(f"[INFO] Output Video:      {output_video}")
    print(f"[INFO] Target Resolution: {target_w}x{target_h} (Fit Mode: {fit_mode})")

    # ===============================
    # HARDWARE DETECTION
    # ===============================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu = True
        print("[INFO] Hardware: NVIDIA GPU detected (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_gpu = True
        print("[INFO] Hardware: Apple Silicon GPU detected (MPS)")
    else:
        device = torch.device("cpu")
        use_gpu = False
        print("[INFO] Hardware: CPU (OpenCV FSRCNN fallback)")

    # ===============================
    # LOAD UPSCALER
    # ===============================
    if use_gpu:
        try:
            from RealESRGAN import RealESRGAN
        except ImportError:
            try:
                from realesrgan import RealESRGAN
            except ImportError:
                print(
                    "[ERROR] RealESRGAN package not found. Run: pip install git+https://github.com/ai-forever/Real-ESRGAN.git"
                )
                exit(1)

        upscaler = RealESRGAN(device, scale=UPSCALE_FACTOR_GPU)

        weights_dir = os.path.abspath("weights")
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, "RealESRGAN_x4.pth")
        if not os.path.exists(weights_path) and os.path.exists("RealESRGAN_x4.pth"):
            weights_path = os.path.abspath("RealESRGAN_x4.pth")
        elif not os.path.exists(weights_path) and os.path.exists("RealESRGAN_x4plus.pth"):
            weights_path = os.path.abspath("RealESRGAN_x4plus.pth")

        upscaler.load_weights(weights_path, download=True)
        print(f"[INFO] Model: Real-ESRGAN x{UPSCALE_FACTOR_GPU}")
    else:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = "models/FSRCNN_x3.pb"
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            print("Please download it from: https://github.com/Saafke/FSRCNN_Tensorflow")
            exit(1)
        sr.readModel(model_path)
        sr.setModel("fsrcnn", UPSCALE_FACTOR_CPU)
        print(f"[INFO] Model: OpenCV FSRCNN x{UPSCALE_FACTOR_CPU}")

    # ===============================
    # FFmpeg: Extract Audio
    # ===============================
    temp_audio = f"temp_audio_{os.getpid()}.aac"
    print("[INFO] Extracting audio track...")
    try:
        ffmpeg.input(input_video).output(
            temp_audio, acodec="copy", vn=None
        ).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print(f"[WARNING] Could not extract audio (no audio track or unsupported codec): {e}")
        temp_audio = None

    # ===============================
    # Video Metadata
    # ===============================
    probe = ffmpeg.probe(input_video)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

    fps = eval(video_stream["r_frame_rate"])
    in_w = int(video_stream["width"])
    in_h = int(video_stream["height"])
    total_frames = int(video_stream.get("nb_frames") or 0)

    if total_frames == 0:
        duration = float(video_stream.get("duration", 0))
        if duration > 0:
            total_frames = int(duration * fps)

    print(f"[INFO] Source Details:    {in_w}x{in_h} @ {fps:.2f} fps ({total_frames} frames)")
    print("-" * 55)

    # ===============================
    # FFmpeg OUTPUT PIPE
    # ===============================
    output_args = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "r": fps,
        "preset": "medium",
        "crf": 18,
    }

    if temp_audio and os.path.exists(temp_audio):
        output_args["acodec"] = "copy"
        input_audio = ffmpeg.input(temp_audio)
        video_input = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{target_w}x{target_h}",
            r=fps,
        )
        process = (
            ffmpeg.output(video_input, input_audio, output_video, **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    else:
        process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{target_w}x{target_h}",
                r=fps,
            )
            .output(output_video, **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    # ===============================
    # PROCESS VIDEO
    # ===============================
    cap = cv2.VideoCapture(input_video)

    try:
        with tqdm(total=total_frames, desc=f"Upscaling to {target_w}x{target_h}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # -----------------------
                # SUPER-RESOLUTION
                # -----------------------
                if use_gpu:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sr_frame = upscaler.predict(frame_rgb)
                    sr_frame = np.array(sr_frame)
                    sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
                else:
                    sr_frame = sr.upsample(frame)

                # -----------------------
                # RESIZE & FIT TO TARGET RESOLUTION
                # -----------------------
                final_frame = resize_and_fit(sr_frame, target_w, target_h, fit_mode=fit_mode)

                # -----------------------
                # WRITE TO PIPE
                # -----------------------
                process.stdin.write(final_frame.astype(np.uint8).tobytes())
                pbar.update(1)
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
        process.wait()

        # ===============================
        # CLEANUP
        # ===============================
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)

    print("-" * 55)
    print(f"🎉 [DONE] Video successfully upscaled to {target_w}x{target_h}!")
    print(f"📁 Output file saved: {output_video}")
    print("=" * 55)


if __name__ == "__main__":
    main()

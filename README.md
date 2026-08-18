# 🎥 AI Video Enhancer & Upscaler

A **local-first, privacy-friendly AI video upscaling tool** built in Python.  
Upscale videos to **1080p (or any target resolution)** using **Real-ESRGAN (AI)** with automatic fallback to **FSRCNN (CPU)**.

No cloud. No uploads. Fully offline.

---

## ✨ Features

### AI Upscaling
- Real-ESRGAN for high-quality AI enhancement  
- NVIDIA CUDA support  
- Apple Silicon (MPS) support  

### Smart Fallback
- Automatically switches to OpenCV **FSRCNN**  
- Works on CPU-only systems without configuration  

### Single-Pass Direct Delivery
- AI upscale directly to your desired resolution (default: 1080p)
- High-quality **Lanczos interpolation** for aspect ratio fitting  

### Aspect Ratio Handling
- Automatic **4:3 → 16:9** correction  
- Smart padding (no stretching or cropping)  

### Audio Preservation
- Extracts original audio  
- Lossless remux into upscaled video  

### Local & Offline
- No uploads  
- No server-side processing  
- No data leaves your machine  

---

## 🖥 Platform & Hardware Support

### Operating Systems
- macOS (Intel & Apple Silicon)  
- Linux  
- Windows  

### Acceleration Backends

| Hardware | Backend | Status |
|--------|--------|--------|
| NVIDIA GPU | CUDA | Real-ESRGAN |
| Apple Silicon (M1–M3) | MPS | Real-ESRGAN |
| CPU-only | OpenCV FSRCNN | Fallback |

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**  
- **FFmpeg** (must be available in PATH)  

---

### Install FFmpeg

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt install ffmpeg
```

#### Windows
Download from: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)  
Ensure `ffmpeg` is added to PATH.

---

## 📦 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/pratik227/video-enhance.git
   cd video-enhance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights**

   **AI Upscaling (Recommended)**  
   Place one of the following files in the project root:
   - `RealESRGAN_x4plus.pth`
   - `RealESRGAN_x4.pth`

   **CPU Fallback**  
   Place this file in the `models/` directory:
   - `FSRCNN_x3.pb`

---

## 🛠 Usage

You can pass the video path and target resolution directly from the terminal command line. The default resolution is **1920x1080** (1080p).

### Basic Command (Default: 1920x1080)
```bash
python upscale_video.py path/to/your_video.mp4
```

### Custom Resolution
Specify any target resolution (e.g. `1920x1080`, `1280x720`, `3840x2160`, or presets like `1080p`, `720p`, `4k`):
```bash
python upscale_video.py path/to/your_video.mp4 1920x1080
python upscale_video.py path/to/your_video.mp4 4k
```

### Custom Output Path
```bash
python upscale_video.py path/to/your_video.mp4 1920x1080 -o output.mp4
```

### Using Flags
```bash
python upscale_video.py -i input.mp4 -r 1920x1080 -o output.mp4
```

### Output
- Video upscaled directly to your target resolution (default 1920x1080)
- Original audio preserved and losslessly muxed
- Automatic aspect ratio handling (pillarbox/letterbox with `--fit-mode pad`, or `stretch`/`crop`)

---

## 📚 Python Dependencies
- `torch`
- `opencv-python`
- `ffmpeg-python`
- `numpy`
- `tqdm`
- `realesrgan`
- `ai-forever Real-ESRGAN`

---

## ⚙️ Processing Notes
- GPU acceleration is auto-detected at runtime
- FSRCNN fallback is used automatically if GPU is unavailable
- Optimized for long-form and archival video processing

### Suitable for
- Old SD footage
- Personal media restoration
- Content remastering
- Offline processing pipelines

---

## 💖 Support & Sponsorship
If this project helps you, consider supporting its development.

💝 **GitHub Sponsors**  
[https://github.com/sponsors/pratik227](https://github.com/sponsors/pratik227)

☕ **Buy Me a Coffee**  
[https://buymeacoffee.com/pratik227](https://buymeacoffee.com/pratik227)

⭐ **Free Support**  
- Star the repository
- Share on X / Reddit / Hacker News
- Recommend it to others

---

## 🧑‍💻 Maintainer
**Pratik Patel**  
Independent builder focused on local-first, privacy-respecting tools

GitHub: [https://github.com/pratik227](https://github.com/pratik227)

---

## 📄 License
MIT License  
See the [LICENSE](LICENSE) file for details.

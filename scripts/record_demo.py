"""Record a deterministic ReelSync demo workflow and assemble an optimized GIF.

Captures frames from demo/index.html?record=1 using Playwright (headless Chromium),
then encodes docs/reelsync-demo.gif with the bundled ffmpeg (imageio_ffmpeg).

Usage:
    python scripts/record_demo.py [--fps 10] [--duration 24] [--scale 1066] [--out docs/reelsync-demo.gif]
"""

import argparse
import pathlib
import shutil
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".venv" / "Lib" / "site-packages"))


def find_chromium():
    candidates = [
        pathlib.Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        pathlib.Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        pathlib.Path.home() / "AppData/Local/Google/Chrome/Application/chrome.exe",
        pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        pathlib.Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--duration", type=float, default=23.5)
    ap.add_argument("--scale", type=int, default=1066, help="output width; height derived 16:9")
    ap.add_argument("--out", default="docs/reelsync-demo.gif")
    ap.add_argument("--dry", action="store_true", help="only smoke-test the page load")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: uv add --dev playwright")
        sys.exit(1)

    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG = shutil.which("ffmpeg")

    if not FFMPEG:
        print("no ffmpeg available")
        sys.exit(1)

    chrome = find_chromium()
    if not chrome:
        print("no chrome/edge found")
        sys.exit(1)

    demo = REPO / "demo" / "index.html"
    url = "file:///" + str(demo).replace("\\", "/") + "?record=1"

    width = 1280
    height = int(width * 9 / 16)  # 720

    with tempfile.TemporaryDirectory(prefix="reelsync_demo_") as td:
        frames = pathlib.Path(td)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chrome, headless=True)
            page = browser.new_page(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
            )
            page.goto(url, wait_until="load")
            page.wait_for_timeout(600)

            if args.dry:
                print("page loaded ok:", page.title())
                browser.close()
                return

            n_frames = int(args.duration * args.fps)
            print(f"recording {n_frames} frames @ {args.fps}fps for {args.duration}s ...")
            for i in range(n_frames):
                page.screenshot(path=str(frames / f"{i:04d}.png"))
                page.wait_for_timeout(1000 // args.fps)
            browser.close()

        import subprocess

        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        height_out = int(args.scale * 9 / 16)
        # ensure even dimensions for x264-free GIF (palette needs even? not strictly, but safe)
        height_out -= height_out % 2
        scale_w = args.scale
        scale_w -= scale_w % 2

        filter_complex = (
            f"[0:v]scale={scale_w}:{height_out}:flags=bilinear,"
            f"split[a][b];"
            f"[a]palettegen=max_colors=256:stats_mode=diff[pal];"
            f"[b][pal]paletteuse=dither=bayer:bayer_scale=2:diff_mode=rectangle"
        )
        cmd = [
            FFMPEG, "-y", "-framerate", str(args.fps),
            "-i", str(frames / "%04d.png"),
            "-filter_complex", filter_complex,
            "-loop", "0",
            str(out),
        ]
        print("assembling GIF:", " ".join(cmd[:6]), "...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("ffmpeg failed:\n", r.stderr[-2000:])
            sys.exit(1)

        size_kb = out.stat().st_size / 1024
        print(f"wrote {out} ({size_kb:.0f} KB, {args.duration}s, {scale_w}x{height_out} @ {args.fps}fps)")


if __name__ == "__main__":
    main()
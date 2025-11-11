#!/usr/bin/env python3
"""
plot_waveform.py

Hiển thị dạng sóng (waveform) 1 chiều của một file âm thanh.
Ví dụ:
    python plot_waveform.py path/to/audio.wav --output waveform.png
"""

import argparse
import os

import librosa
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vẽ waveform 1 chiều của file âm thanh.")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Đường dẫn tới file âm thanh cần xem.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Sample rate mong muốn khi load (mặc định giữ nguyên sample rate gốc).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn lưu ảnh waveform (PNG). Nếu bỏ trống sẽ hiển thị trực tiếp.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio_path

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {audio_path}")

    # Load audio; librosa trả về mảng numpy 1 chiều (mono). Nếu file stereo sẽ tự mix về mono.
    waveform, sample_rate = librosa.load(audio_path, sr=args.sample_rate, mono=True)
    duration = waveform.shape[0] / sample_rate
    time_axis = librosa.times_like(waveform, sr=sample_rate)

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, waveform, linewidth=1)
    plt.title(f"Waveform - {os.path.basename(audio_path)}")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Biên độ")
    plt.tight_layout()

    if args.output:
        output_path = args.output
        plt.savefig(output_path, dpi=150)
        print(f"Đã lưu waveform vào: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

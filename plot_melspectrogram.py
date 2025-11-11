#!/usr/bin/env python3
"""
plot_melspectrogram.py

Chuyển một file âm thanh sang Mel spectrogram và hiển thị/lưu hình ảnh.
Ví dụ:
    python plot_melspectrogram.py input.wav --output mel.png
"""

import argparse
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vẽ Mel spectrogram của file âm thanh.")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Đường dẫn tới file âm thanh cần chuyển.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate sử dụng khi load audio (mặc định 22.05 kHz)."
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=128,
        help="Số lượng Mel bands (mặc định 128)."
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048,
        help="Kích thước FFT (mặc định 2048)."
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length giữa các frame (mặc định 512)."
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=0.0,
        help="Tần số thấp nhất cho Mel scale (Hz)."
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Tần số cao nhất cho Mel scale (Hz). Mặc định None = sample_rate/2."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn lưu ảnh Mel spectrogram (PNG). Nếu bỏ trống sẽ hiển thị trực tiếp."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.audio_path):
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {args.audio_path}")

    # Load audio và chuyển sang mono.
    waveform, sample_rate = librosa.load(
        args.audio_path,
        sr=args.sample_rate,
        mono=True,
        res_type="kaiser_fast"
    )

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax
    )

    # Chuyển sang dB để dễ quan sát
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        hop_length=args.hop_length,
        x_axis='time',
        y_axis='mel',
        fmin=args.fmin,
        fmax=args.fmax
    )
    plt.title(f"Mel Spectrogram - {os.path.basename(args.audio_path)}")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"Đã lưu Mel spectrogram vào: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

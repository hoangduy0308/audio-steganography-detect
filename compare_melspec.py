#!/usr/bin/env python3
"""
compare_melspec.py

So sánh Mel spectrogram giữa hai file âm thanh (ví dụ bản sạch và bản stego).
Xuất các ảnh riêng lẻ và ảnh chênh lệch tuyệt đối để giúp nhận biết khác biệt.

Ví dụ:
    python compare_melspec.py clean.wav stego.wav --output-dir out_compare
"""

import argparse
import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh Mel spectrogram của hai file âm thanh.")
    parser.add_argument("clean_path", type=str, help="Đường dẫn file âm thanh sạch.")
    parser.add_argument("stego_path", type=str, help="Đường dẫn file âm thanh nhúng (stego).")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate dùng khi load audio (mặc định 22.05 kHz).",
    )
    parser.add_argument("--n-mels", type=int, default=128, help="Số lượng Mel bands.")
    parser.add_argument("--n-fft", type=int, default=2048, help="Kích thước FFT.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="compare_melspec_output",
        help="Thư mục lưu kết quả so sánh (ảnh PNG).",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=0.0,
        help="Tần số thấp nhất cho Mel scale (Hz).",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help="Tần số cao nhất cho Mel scale (Hz). (None = sr/2).",
    )
    return parser.parse_args()


def load_mel_spectrogram(
    audio_path: str,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmin: float,
    fmax: float | None,
) -> np.ndarray:
    waveform, sr_actual = librosa.load(audio_path, sr=sr, mono=True, res_type="kaiser_fast")
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr_actual,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def plot_and_save_spec(data: np.ndarray, sr: int, hop_length: int, title: str, outfile: Path, fmin: float, fmax: float | None) -> None:
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        data,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
    )
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Đã lưu {outfile}")


def main() -> None:
    args = parse_args()

    clean_path = Path(args.clean_path)
    stego_path = Path(args.stego_path)
    if not clean_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file sạch: {clean_path}")
    if not stego_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file stego: {stego_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mel_clean = load_mel_spectrogram(
        str(clean_path),
        sr=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    mel_stego = load_mel_spectrogram(
        str(stego_path),
        sr=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    # Đảm bảo kích thước giống nhau (pad/trim nếu cần)
    min_time_bins = min(mel_clean.shape[1], mel_stego.shape[1])
    mel_clean = mel_clean[:, :min_time_bins]
    mel_stego = mel_stego[:, :min_time_bins]

    diff_abs = np.abs(mel_stego - mel_clean)

    plot_and_save_spec(
        mel_clean,
        args.sample_rate,
        args.hop_length,
        f"Mel Spectrogram (Clean) - {clean_path.name}",
        output_dir / f"{clean_path.stem}_mel.png",
        args.fmin,
        args.fmax,
    )
    plot_and_save_spec(
        mel_stego,
        args.sample_rate,
        args.hop_length,
        f"Mel Spectrogram (Stego) - {stego_path.name}",
        output_dir / f"{stego_path.stem}_mel.png",
        args.fmin,
        args.fmax,
    )
    plot_and_save_spec(
        diff_abs,
        args.sample_rate,
        args.hop_length,
        "Chênh lệch |Mel_stego - Mel_clean| (dB)",
        output_dir / "difference_abs.png",
        args.fmin,
        args.fmax,
    )

    # Thống kê nhanh các chênh lệch lớn
    mean_diff = float(diff_abs.mean())
    max_diff = float(diff_abs.max())
    print(f"Độ lệch trung bình (dB): {mean_diff:.4f}")
    print(f"Độ lệch lớn nhất (dB): {max_diff:.4f}")
    print(f"Đã lưu kết quả tại: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

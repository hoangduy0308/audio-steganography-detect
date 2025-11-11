#!/usr/bin/env python3
"""GUI kiểm tra audio steganography bằng mô hình đã huấn luyện."""

import argparse
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# --- Tham số tiền xử lý khớp với notebook huấn luyện ---
SAMPLE_RATE = 22_050
FIXED_DURATION_SEC = 4.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
EXPECTED_SPECTROGRAM_COLS = int(np.ceil(FIXED_DURATION_SEC * SAMPLE_RATE / HOP_LENGTH))
DECISION_THRESHOLD = 0.2939  # Ngưỡng F1 tối ưu từ run tốt nhất

CLEAN_LABEL = "Không nhúng"
STEGO_LABEL = "Đã nhúng"


def resolve_model_path(user_path: Optional[str]) -> Path:
    """Xác định đường dẫn mô hình hợp lệ."""
    if user_path:
        model_path = Path(user_path).expanduser()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")

    candidates = [
        Path("D:/Work/TTNT/audio steganography/models/best_model_audio_steganalysis_v1.keras"),
        Path("/mnt/d/Work/TTNT/audio steganography/models/best_model_audio_steganalysis_v1.keras"),
        Path.cwd() / "models" / "best_model_audio_steganalysis_v1.keras",
        Path.cwd() / "best_model_audio_steganalysis_v1.keras",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy mô hình. Vui lòng truyền tham số --model tới file '.keras' tương ứng."
    )


def list_training_files(data_root: Optional[Path]) -> Tuple[Path, ...]:
    """Lấy danh sách file train để tính thống kê chuẩn hóa."""
    if not data_root:
        return tuple()
    folders = [
        data_root / "clean" / "train",
        data_root / "stego" / "train",
    ]
    files = []
    for folder in folders:
        if folder.exists():
            files.extend(sorted(folder.glob("*.wav")))
    return tuple(files)


def load_and_preprocess_audio(
    file_path: Path,
    sr: int = SAMPLE_RATE,
    duration: float = FIXED_DURATION_SEC,
) -> Optional[np.ndarray]:
    """Tạo log-mel + delta + delta2 spectrogram 3 kênh cho 1 file âm thanh."""
    try:
        samples, _ = librosa.load(
            file_path,
            sr=sr,
            duration=duration,
            res_type="kaiser_fast",
        )
        target_length = int(duration * sr)
        if len(samples) < target_length:
            samples = librosa.util.fix_length(samples, size=target_length)

        mel_spec = librosa.feature.melspectrogram(
            y=samples,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        mel_spec = np.maximum(1e-10, mel_spec)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        if log_mel.shape[1] < EXPECTED_SPECTROGRAM_COLS:
            pad_width = EXPECTED_SPECTROGRAM_COLS - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
        elif log_mel.shape[1] > EXPECTED_SPECTROGRAM_COLS:
            log_mel = log_mel[:, :EXPECTED_SPECTROGRAM_COLS]

        delta_1 = librosa.feature.delta(log_mel, order=1, mode="nearest")
        delta_2 = librosa.feature.delta(log_mel, order=2, mode="nearest")

        stacked = np.stack([log_mel, delta_1, delta_2], axis=-1)
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
        return stacked.astype(np.float32, copy=False)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[LỖI] Không xử lý được file {file_path}: {exc}", file=sys.stderr)
        return None


def compute_normalization_stats(
    files: Tuple[Path, ...],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Tính mean/std per-channel trên tập train."""
    if not files:
        return None

    sum_channels = np.zeros(3, dtype=np.float64)
    sum_sq_channels = np.zeros(3, dtype=np.float64)
    total_frames = 0

    for idx, audio_path in enumerate(files, start=1):
        spec = load_and_preprocess_audio(audio_path)
        if spec is None:
            continue
        spec64 = spec.astype(np.float64)
        sum_channels += spec64.sum(axis=(0, 1))
        sum_sq_channels += np.square(spec64).sum(axis=(0, 1))
        total_frames += spec.shape[0] * spec.shape[1]
        if idx % 200 == 0 or idx == len(files):
            print(f"[INFO] Đã xử lý {idx}/{len(files)} file để tính thống kê.")

    if total_frames == 0:
        return None

    mean = sum_channels / total_frames
    variance = np.maximum(sum_sq_channels / total_frames - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def load_or_build_stats(
    stats_path: Path,
    data_root: Optional[Path],
) -> Optional[dict]:
    """Tải hoặc tính thống kê chuẩn hóa."""
    if stats_path.exists():
        data = np.load(stats_path)
        return {
            "mean": data["mean"],
            "std": data["std"],
            "source": data.get("source", "file"),
        }

    files = list_training_files(data_root)
    if not files:
        print("[CẢNH BÁO] Không tìm thấy stats và không có dữ liệu train để tính toán.")
        return None

    print("[INFO] Đang tính toán thống kê chuẩn hóa từ tập train...")
    stats = compute_normalization_stats(files)
    if stats is None:
        print("[CẢNH BÁO] Không thể tính được thống kê chuẩn hóa.")
        return None

    mean, std = stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_path,
        mean=mean,
        std=std,
        source=str(data_root.resolve()) if data_root else "",
    )
    print(f"[INFO] Đã lưu thống kê chuẩn hóa tại: {stats_path}")
    return {"mean": mean, "std": std, "source": str(data_root.resolve())}


def normalize_spectrogram(
    spec: np.ndarray,
    stats: Optional[dict],
) -> np.ndarray:
    """Chuẩn hóa spectrogram với mean/std theo từng kênh."""
    if stats is not None:
        mean = stats["mean"].reshape(1, 1, -1)
        std = stats["std"].reshape(1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        return (spec - mean) / std

    channel_mean = spec.mean(axis=(0, 1), keepdims=True)
    channel_std = spec.std(axis=(0, 1), keepdims=True)
    channel_std = np.where(channel_std == 0, 1.0, channel_std)
    print("[CẢNH BÁO] Đang sử dụng chuẩn hóa theo từng file (không có stats tập train).")
    return (spec - channel_mean) / channel_std


def predict_probability(
    model: tf.keras.Model,
    audio_path: Path,
    stats: Optional[dict],
) -> float:
    """Trả về xác suất lớp stego."""
    spec = load_and_preprocess_audio(audio_path)
    if spec is None:
        raise ValueError("Không thể tiền xử lý file âm thanh.")
    norm_spec = normalize_spectrogram(spec, stats)
    sample = np.expand_dims(norm_spec, axis=0)
    prob = model.predict(sample, verbose=0)[0][0]
    return float(prob)


class AudioStegGUI:
    """Lớp GUI chính."""

    def __init__(
        self,
        model: tf.keras.Model,
        stats: Optional[dict],
        model_path: Path,
    ) -> None:
        self.model = model
        self.stats = stats
        self.model_path = model_path
        self.current_audio: Optional[Path] = None

        self.root = tk.Tk()
        self.root.title("Phát hiện Audio Steganography")
        self.root.geometry("640x300")
        self.root.resizable(False, False)

        self.file_var = tk.StringVar(value="Chưa chọn file.")
        self.result_var = tk.StringVar(value="Kết quả sẽ hiển thị tại đây.")

        self._build_widgets()

        if stats is None:
            messagebox.showwarning(
                "Thiếu thống kê chuẩn hóa",
                (
                    "Không tìm thấy thống kê mean/std của tập train.\n"
                    "Chương trình sẽ dùng chuẩn hóa theo từng file, có thể làm giảm độ chính xác.\n"
                    "Hãy cung cấp --data-root để tạo thống kê chuẩn nếu có thể."
                ),
            )

    def _build_widgets(self) -> None:
        padding = {"padx": 12, "pady": 8}

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, **padding)

        ttk.Label(frame, text=f"Mô hình: {self.model_path}").pack(
            anchor=tk.W, padx=4, pady=(0, 12)
        )

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Button(
            btn_frame,
            text="Chọn file WAV",
            command=self.select_file,
        ).pack(side=tk.LEFT)

        self.analyze_btn = ttk.Button(
            btn_frame,
            text="Phân tích",
            command=self.analyze,
            state=tk.DISABLED,
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(frame, textvariable=self.file_var, foreground="#555555").pack(
            anchor=tk.W, padx=4
        )

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(frame, text="Kết quả:").pack(anchor=tk.W, padx=4)
        self.result_label = ttk.Label(
            frame,
            textvariable=self.result_var,
            justify=tk.LEFT,
            font=("Arial", 12, "bold"),
        )
        self.result_label.pack(anchor=tk.W, padx=4)

    def select_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Chọn file WAV",
            filetypes=[("WAV files", "*.wav")],
        )
        if not file_path:
            return
        self.current_audio = Path(file_path)
        self.file_var.set(f"Đã chọn: {self.current_audio}")
        self.result_var.set("Nhấn 'Phân tích' để dự đoán.")
        self.result_label.configure(foreground="#000000")
        self.analyze_btn.config(state=tk.NORMAL)

    def analyze(self) -> None:
        if not self.current_audio:
            messagebox.showinfo("Chưa có file", "Hãy chọn một file WAV trước.")
            return

        self.analyze_btn.config(state=tk.DISABLED)
        self.result_var.set("Đang phân tích...")
        self.root.update_idletasks()

        def worker() -> None:
            try:
                prob = predict_probability(self.model, self.current_audio, self.stats)
            except Exception as exc:  # pylint: disable=broad-except
                message = f"Lỗi khi phân tích file:\n{exc}"
                self.root.after(
                    0,
                    lambda: messagebox.showerror("Lỗi", message),
                )
                self.root.after(
                    0,
                    lambda: [
                        self.result_var.set("Không thể phân tích file."),
                        self.result_label.configure(foreground="#FF0000"),
                        self.analyze_btn.config(state=tk.NORMAL),
                    ],
                )
                return

            label = STEGO_LABEL if prob >= DECISION_THRESHOLD else CLEAN_LABEL
            color = "#C00000" if label == STEGO_LABEL else "#007A2E"
            text = (
                f"Xác suất stego: {prob:.4f}\n"
                f"Ngưỡng so sánh: {DECISION_THRESHOLD:.4f}\n"
                f"Dự đoán: {label}"
            )
            self.root.after(
                0,
                lambda: [
                    self.result_var.set(text),
                    self.result_label.configure(foreground=color),
                    self.analyze_btn.config(state=tk.NORMAL),
                ],
            )

        threading.Thread(target=worker, daemon=True).start()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GUI kiểm tra file audio đã nhúng stego hay chưa."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Đường dẫn tới file .keras (nếu bỏ trống sẽ tự dò).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Thư mục chứa dữ liệu train (clean/stego/train) để tính thống kê chuẩn hóa.",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Đường dẫn file .npz lưu mean/std (sẽ sinh tự động nếu thiếu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model_path = resolve_model_path(args.model)
    except FileNotFoundError as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Đang nạp mô hình từ: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("[INFO] Nạp mô hình thành công.")

    stats_path = (
        Path(args.stats).expanduser()
        if args.stats
        else model_path.with_name("audio_normalization_stats.npz")
    )
    data_root = Path(args.data_root).expanduser() if args.data_root else None
    stats = load_or_build_stats(stats_path, data_root)

    app = AudioStegGUI(model=model, stats=stats, model_path=model_path)
    app.run()


if __name__ == "__main__":
    main()

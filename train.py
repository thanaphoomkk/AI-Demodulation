"""
train.py
========
เทรนโมเดล CNN+LSTM Demodulator

วิธีใช้:
    python train.py                 # ค่าเริ่มต้น: QPSK
    python train.py --mod bpsk
    python train.py --mod qam16 --epochs 30

ผลลัพธ์:
    - บันทึกโมเดลไว้ที่  models/<mod>_demod.keras
    - บันทึกกราฟ loss / accuracy / BER ไว้ที่ results/<mod>_training.png
"""

from __future__ import annotations
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from generate_signal import generate_dataset, BITS_PER_SYMBOL
from model import build_model

MODELS_DIR = "models"
RESULTS_DIR = "results"


def plot_history(history, mod: str, out_path: str) -> None:
    """วาดกราฟ loss, accuracy และ BER ต่อ epoch"""
    h = history.history
    epochs = range(1, len(h["loss"]) + 1)

    # BER = 1 - binary_accuracy
    train_ber = 1.0 - np.array(h["binary_accuracy"])
    val_ber = 1.0 - np.array(h["val_binary_accuracy"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ----- Loss -----
    axes[0].plot(epochs, h["loss"], "b-o", label="train")
    axes[0].plot(epochs, h["val_loss"], "r-s", label="val")
    axes[0].set_title("Loss (Binary Crossentropy)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ----- Accuracy -----
    axes[1].plot(epochs, h["binary_accuracy"], "b-o", label="train")
    axes[1].plot(epochs, h["val_binary_accuracy"], "r-s", label="val")
    axes[1].set_title("Bit Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ----- BER (log scale) -----
    axes[2].semilogy(epochs, train_ber + 1e-6, "b-o", label="train")
    axes[2].semilogy(epochs, val_ber + 1e-6, "r-s", label="val")
    axes[2].set_title("Bit Error Rate (BER)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("BER (log)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Training History - {mod.upper()}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[บันทึกกราฟ] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="เทรน CNN+LSTM Demodulator")
    parser.add_argument("--mod", default="qpsk",
                        choices=list(BITS_PER_SYMBOL.keys()),
                        help="ชนิด modulation ที่จะเทรน")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--n_train", type=int, default=4000)
    parser.add_argument("--n_val", type=int, default=800)
    parser.add_argument("--symbols", type=int, default=64,
                        help="จำนวนสัญลักษณ์ต่อตัวอย่าง")
    parser.add_argument("--sps", type=int, default=8,
                        help="samples ต่อสัญลักษณ์")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bps = BITS_PER_SYMBOL[args.mod]
    print("=" * 60)
    print(f" เทรน Demodulator: {args.mod.upper()} ({bps} บิต/สัญลักษณ์)")
    print("=" * 60)

    # ---------- สร้างชุดข้อมูล ----------
    print("กำลังสร้างชุดข้อมูล train/val ...")
    X_train, y_train = generate_dataset(
        args.mod, n_examples=args.n_train,
        symbols_per_example=args.symbols, sps=args.sps,
        snr_range=(-5, 25), seed=args.seed,
    )
    X_val, y_val = generate_dataset(
        args.mod, n_examples=args.n_val,
        symbols_per_example=args.symbols, sps=args.sps,
        snr_range=(-5, 25), seed=args.seed + 1,
    )
    print(f"  X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  X_val  ={X_val.shape}, y_val  ={y_val.shape}")

    # ---------- สร้างและเทรนโมเดล ----------
    model = build_model(symbols_per_example=args.symbols,
                        sps=args.sps, bits_per_symbol=bps)
    model.summary()

    model_path = os.path.join(MODELS_DIR, f"{args.mod}_demod.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_binary_accuracy",
            save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy", patience=6,
            restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(model_path)
    print(f"[บันทึกโมเดล] {model_path}")

    # ---------- สรุปผล + วาดกราฟ ----------
    val_acc = history.history["val_binary_accuracy"][-1]
    val_ber = 1.0 - val_acc
    print("-" * 60)
    print(f" ผลบน validation:  Accuracy = {val_acc*100:.2f}%  |  BER = {val_ber:.4e}")
    print("-" * 60)

    plot_path = os.path.join(RESULTS_DIR, f"{args.mod}_training.png")
    plot_history(history, args.mod, plot_path)


if __name__ == "__main__":
    main()

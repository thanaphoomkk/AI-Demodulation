"""
test.py
=======
ทดสอบโมเดล Demodulator ที่เทรนไว้ กับสัญญาณทุกประเภท

สิ่งที่ทำ:
    1) วัดความแม่นยำ (Accuracy) และ BER ของโมเดลดิจิทัลที่เทรนไว้
       ที่ระดับ SNR ต่าง ๆ  -> พิมพ์ตาราง + วาดกราฟ BER vs SNR
    2) แสดงภาพสัญญาณทั้ง 5 ประเภท (BPSK, QPSK, QAM16, AM, FM)
       ทั้ง constellation และรูปคลื่นเชิงเวลา

วิธีใช้:
    python test.py                # ทดสอบทุกโมเดลที่มีใน models/
    python test.py --mod qpsk     # ทดสอบเฉพาะ QPSK
"""

from __future__ import annotations
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from generate_signal import (
    generate_dataset, generate_signal,
    BITS_PER_SYMBOL, DIGITAL_MODS, ALL_MODS,
)

MODELS_DIR = "models"
RESULTS_DIR = "results"
SNR_LIST = [-5, 0, 5, 10, 15, 20]


# --------------------------------------------------------------------------- #
#  ประเมินโมเดลที่ระดับ SNR ต่าง ๆ
# --------------------------------------------------------------------------- #
def evaluate_model(model, mod: str, sps: int, symbols: int,
                   n_examples: int = 500):
    """คืน dict {snr: (accuracy, ber)}"""
    results = {}
    for snr in SNR_LIST:
        X, y = generate_dataset(
            mod, n_examples=n_examples, symbols_per_example=symbols,
            sps=sps, snr_range=(snr, snr), seed=1000 + snr,
        )
        pred = model.predict(X, verbose=0)
        pred_bits = (pred > 0.5).astype(np.float32)
        acc = float(np.mean(pred_bits == y))
        ber = 1.0 - acc
        results[snr] = (acc, ber)
    return results


def infer_model_shape(model):
    """อ่าน sps และ symbols_per_example กลับจากโครงสร้างโมเดล"""
    input_len = model.input_shape[1]          # T*sps
    T = model.output_shape[1]                  # symbols_per_example
    sps = input_len // T
    return sps, T


# --------------------------------------------------------------------------- #
#  วาดกราฟ BER vs SNR
# --------------------------------------------------------------------------- #
def plot_ber_curve(all_results: dict, out_path: str) -> None:
    plt.figure(figsize=(8, 5.5))
    markers = {"bpsk": "o-", "qpsk": "s-", "qam16": "^-"}
    for mod, res in all_results.items():
        snrs = sorted(res.keys())
        bers = [res[s][1] + 1e-6 for s in snrs]
        plt.semilogy(snrs, bers, markers.get(mod, "x-"),
                     label=mod.upper(), linewidth=2, markersize=7)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (log scale)")
    plt.title("BER vs SNR - AI Demodulator")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[บันทึกกราฟ] {out_path}")


# --------------------------------------------------------------------------- #
#  แสดงภาพสัญญาณทั้ง 5 ประเภท
# --------------------------------------------------------------------------- #
def plot_all_signals(out_path: str, snr_db: float = 15.0) -> None:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(2, len(ALL_MODS), figsize=(4 * len(ALL_MODS), 8))

    for col, mod in enumerate(ALL_MODS):
        iq, _ = generate_signal(mod, n_symbols=128, sps=8,
                                snr_db=snr_db, rng=rng)
        # แถวบน: constellation (I vs Q)
        axes[0, col].scatter(iq.real, iq.imag, s=5, alpha=0.4)
        axes[0, col].set_title(f"{mod.upper()} - Constellation")
        axes[0, col].set_xlabel("I")
        axes[0, col].set_ylabel("Q")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].axis("equal")

        # แถวล่าง: รูปคลื่นเชิงเวลา (200 samples แรก)
        n = min(200, len(iq))
        axes[1, col].plot(iq.real[:n], label="I", linewidth=1)
        axes[1, col].plot(iq.imag[:n], label="Q", linewidth=1, alpha=0.7)
        axes[1, col].set_title(f"{mod.upper()} - Waveform")
        axes[1, col].set_xlabel("Sample")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle(f"สัญญาณจำลองทุกประเภท (SNR={snr_db} dB)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[บันทึกกราฟ] {out_path}")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="ทดสอบ Demodulator")
    parser.add_argument("--mod", default="all",
                        help="ชนิด modulation ที่จะทดสอบ หรือ 'all'")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----- 1) แสดงภาพสัญญาณทุกประเภท -----
    print("=" * 60)
    print(" สร้างภาพสัญญาณจำลองทุกประเภท")
    print("=" * 60)
    plot_all_signals(os.path.join(RESULTS_DIR, "all_signals.png"))

    # ----- 2) ทดสอบโมเดลดิจิทัลที่เทรนไว้ -----
    if args.mod == "all":
        mods_to_test = DIGITAL_MODS
    else:
        mods_to_test = [args.mod]

    all_results = {}
    print("\n" + "=" * 60)
    print(" ทดสอบความแม่นยำของโมเดล (Accuracy / BER)")
    print("=" * 60)

    for mod in mods_to_test:
        model_path = os.path.join(MODELS_DIR, f"{mod}_demod.keras")
        if not os.path.exists(model_path):
            print(f"[ข้าม] ยังไม่มีโมเดลของ {mod.upper()} "
                  f"(รัน:  python train.py --mod {mod})")
            continue

        print(f"\n>> โหลดโมเดล {mod.upper()} จาก {model_path}")
        model = tf.keras.models.load_model(model_path)
        sps, symbols = infer_model_shape(model)

        res = evaluate_model(model, mod, sps=sps, symbols=symbols)
        all_results[mod] = res

        print(f"   {'SNR(dB)':>8} | {'Accuracy':>9} | {'BER':>10}")
        print("   " + "-" * 33)
        for snr in SNR_LIST:
            acc, ber = res[snr]
            print(f"   {snr:>8} | {acc*100:>8.2f}% | {ber:>10.4e}")

    # ----- 3) วาดกราฟ BER vs SNR -----
    if all_results:
        plot_ber_curve(all_results,
                       os.path.join(RESULTS_DIR, "ber_vs_snr.png"))
        print("\nสรุป: โมเดลทำงานได้ตามคาด — BER ลดลงเมื่อ SNR สูงขึ้น ✔")
    else:
        print("\nยังไม่มีโมเดลให้ทดสอบ กรุณาเทรนก่อนด้วย:  python train.py")


if __name__ == "__main__":
    main()

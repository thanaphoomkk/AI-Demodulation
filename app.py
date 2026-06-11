"""
app.py  —  Flask Web App สำหรับ AI Demodulation
วิธีรัน:
    pip install flask numpy scipy matplotlib
    python app.py
แล้วเปิด http://localhost:5000
"""

from __future__ import annotations
import io, base64, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from flask import Flask, render_template, request, jsonify

from generate_signal import generate_signal, ALL_MODS, DIGITAL_MODS
from demod_classical import demodulate, iq_from_arrays

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# ------------------------------------------------------------------ AI loader --

_model_cache: dict = {}

def _load_ai_model(mod: str):
    """โหลดโมเดล AI จากไฟล์ .keras (cache ไว้ในหน่วยความจำ)"""
    if mod in _model_cache:
        return _model_cache[mod]
    path = os.path.join("models", f"{mod}_demod.keras")
    if not os.path.exists(path):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        _model_cache[mod] = model
        return model
    except Exception:
        return None


def _ai_demodulate(iq: np.ndarray, mod: str, sps: int = 8) -> list[int] | None:
    """ถอดรหัสด้วย AI model คืน list[int] หรือ None ถ้าไม่มีโมเดล"""
    model = _load_ai_model(mod)
    if model is None:
        return None

    input_len = model.input_shape[1]   # T * sps

    iq_real = iq.real.astype(np.float32)
    iq_imag = iq.imag.astype(np.float32)

    bits_out: list[int] = []
    i = 0
    while i + input_len <= len(iq):
        chunk = np.stack([iq_real[i:i + input_len],
                          iq_imag[i:i + input_len]], axis=-1)[np.newaxis]
        pred = model.predict(chunk, verbose=0)   # (1, T, bps)
        bits_out.extend((pred[0] > 0.5).astype(int).flatten().tolist())
        i += input_len

    return bits_out if bits_out else None


# ------------------------------------------------------------------ helpers --

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _style_ax(ax):
    ax.set_facecolor("#1e293b")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#93c5fd")
    ax.grid(alpha=0.2, color="#334155")


def _make_plots(iq: np.ndarray, mod: str,
                ai_bits: list[int], classical_bits: list[int],
                src_bits: list[int] | None = None,
                fc: float = 0.1) -> dict:

    n   = min(300, len(iq))
    t   = np.arange(n)
    rf  = iq.real[:n] * np.cos(2*np.pi*fc*t) - iq.imag[:n] * np.sin(2*np.pi*fc*t)

    # ── กราฟ 1: IQ Baseband + RF Waveform ──────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    fig.patch.set_facecolor("#0f172a")
    _style_ax(axes[0]); _style_ax(axes[1])

    axes[0].plot(t, iq.real[:n], color="#60a5fa", lw=1.2, label="I")
    axes[0].plot(t, iq.imag[:n], color="#34d399", lw=1.2, alpha=0.8, label="Q")
    axes[0].set_title(f"IQ Baseband — {mod.upper()}")
    axes[0].legend(fontsize=8, facecolor="#0f172a", labelcolor="#e2e8f0")

    axes[1].plot(t, rf, color="#f59e0b", lw=1.0, label=f"RF  (fc={fc:.2f})")
    axes[1].set_title(f"RF Waveform — {mod.upper()}")
    axes[1].set_xlabel("Sample")
    axes[1].legend(fontsize=8, facecolor="#0f172a", labelcolor="#e2e8f0")
    fig.tight_layout()
    wave_b64 = _fig_to_b64(fig)

    # ── กราฟ 2: Constellation ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor("#0f172a"); _style_ax(ax)
    ax.scatter(iq.real, iq.imag, s=6, alpha=0.4, color="#60a5fa")
    ax.axhline(0, color="#475569", lw=0.5); ax.axvline(0, color="#475569", lw=0.5)
    ax.set_title(f"Constellation — {mod.upper()}")
    ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.axis("equal")
    const_b64 = _fig_to_b64(fig)

    # ── กราฟ 3: บาร์ถูก-ผิด AI vs Classical ──────────────────────────────
    if src_bits:
        show   = min(32, len(src_bits))
        src_s  = src_bits[:show]
        ai_s   = ai_bits[:show]   if ai_bits   else src_s
        cl_s   = classical_bits[:show] if classical_bits else src_s

        ber_ai = sum(a != b for a, b in zip(src_s, ai_s))  / max(show, 1)
        ber_cl = sum(a != b for a, b in zip(src_s, cl_s))  / max(show, 1)

        fig, ax3 = plt.subplots(figsize=(max(9, show*0.4), 3.2))
        fig.patch.set_facecolor("#0f172a"); _style_ax(ax3)
        ax3.set_xlim(-0.5, show - 0.5)
        ax3.set_ylim(-0.5, 2.7)
        ax3.set_yticks([0.3, 1.7])
        ax3.set_yticklabels(["Classical", "AI"])
        ax3.set_xlabel("บิตที่")
        ax3.set_title(
            f"ผลถอดรหัส — {mod.upper()}  |  "
            f"BER(AI)={ber_ai:.3f}  BER(Classical)={ber_cl:.3f}"
        )
        C_OK, C_ERR = "#34d399", "#f87171"
        for i in range(show):
            s = int(src_s[i])
            # AI bar
            ok = int(ai_s[i]) == s
            ax3.bar(i, 0.55, bottom=1.42, width=0.72,
                    color=C_OK if ok else C_ERR, alpha=0.88)
            ax3.text(i, 1.7, str(int(ai_s[i])),
                     ha="center", va="center", fontsize=9,
                     fontweight="bold", color="#0f172a")
            # Classical bar
            ok = int(cl_s[i]) == s
            ax3.bar(i, 0.55, bottom=0.03, width=0.72,
                    color=C_OK if ok else C_ERR, alpha=0.88)
            ax3.text(i, 0.3, str(int(cl_s[i])),
                     ha="center", va="center", fontsize=9,
                     fontweight="bold", color="#0f172a")
            # src label
            ax3.text(i, 2.42, str(s), ha="center", va="center",
                     fontsize=8, color="#f59e0b", fontweight="bold")

        # legend
        pk = mpatches.Patch(color=C_OK,  label="ถูกต้อง")
        pe = mpatches.Patch(color=C_ERR, label="ผิด")
        ax3.legend(handles=[pk, pe], loc="upper right",
                   facecolor="#0f172a", edgecolor="#334155",
                   labelcolor="#e2e8f0", fontsize=8)
        # src label header
        ax3.text(-0.45, 2.42, "src", ha="left", va="center",
                 fontsize=7, color="#94a3b8")
        fig.tight_layout()
        decode_b64 = _fig_to_b64(fig)
    else:
        decode_b64 = None

    return {"waveform":    wave_b64,
            "constellation": const_b64,
            "decode_bar":  decode_b64}


def _ber(src: list[int], dec: list[int]) -> float:
    n = min(len(src), len(dec))
    return sum(a != b for a, b in zip(src[:n], dec[:n])) / max(n, 1)


# ------------------------------------------------------------------ routes --

@app.route("/")
def index():
    return render_template("index.html", mods=ALL_MODS, digital_mods=DIGITAL_MODS)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data  = request.get_json()
    mod   = data.get("mod", "qpsk").lower()
    snr   = float(data.get("snr", 10))
    n_sym = int(data.get("n_symbols", 64))
    fc    = float(data.get("fc", 0.1))
    fc    = max(0.01, min(fc, 0.45))   # clamp ไม่ให้เกิน Nyquist
    sps   = 8

    rng = np.random.default_rng()
    iq, src_bits = generate_signal(mod, n_symbols=n_sym, sps=sps,
                                   snr_db=snr, rng=rng)

    if mod not in DIGITAL_MODS:
        plots = _make_plots(iq, mod, [], [], fc=fc)
        return jsonify({"bits_ai": "(Analog)", "bits_classical": "(Analog)",
                        "bits_source": "(Analog)",
                        "total_bits": 0, "ber_ai": None, "ber_classical": None,
                        "method": "analog", "plots": plots})

    src = list(src_bits)

    # Classical
    classical = demodulate(iq, mod, sps=sps)
    ber_cl = _ber(src, classical)

    # AI (fallback to classical ถ้าไม่มีโมเดล)
    ai_decoded = _ai_demodulate(iq, mod, sps=sps)
    if ai_decoded is not None:
        method = "ai"
        ber_ai = _ber(src, ai_decoded)
    else:
        ai_decoded = classical
        method = "classical_fallback"
        ber_ai = ber_cl

    plots = _make_plots(iq, mod, ai_decoded, classical, src_bits=src, fc=fc)

    return jsonify({
        "bits_source":    "".join(str(b) for b in src[:128]),
        "bits_ai":        "".join(str(b) for b in ai_decoded[:128]),
        "bits_classical": "".join(str(b) for b in classical[:128]),
        "total_bits":     len(classical),
        "ber_ai":         round(ber_ai, 4),
        "ber_classical":  round(ber_cl, 4),
        "method":         method,
        "plots":          plots,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    mod = request.form.get("mod", "qpsk").lower()
    sps = int(request.form.get("sps", 1))
    f   = request.files.get("file")

    if not f:
        return jsonify({"error": "ไม่ได้แนบไฟล์"}), 400

    try:
        txt  = f.read().decode("utf-8")
        rows = [line.strip().split(",") for line in txt.splitlines()
                if line.strip() and not line.startswith("#")]
        arr  = np.array([[float(r[0]), float(r[1])] for r in rows if len(r) >= 2])
        iq   = iq_from_arrays(arr[:, 0], arr[:, 1])
    except Exception as e:
        return jsonify({"error": f"อ่านไฟล์ไม่ได้: {e}"}), 400

    if mod not in DIGITAL_MODS:
        return jsonify({"error": f"อัปโหลดรองรับเฉพาะ {DIGITAL_MODS}"}), 400

    classical  = demodulate(iq, mod, sps=sps)
    ai_decoded = _ai_demodulate(iq, mod, sps=sps)
    if ai_decoded is not None:
        method = "ai"
    else:
        ai_decoded = classical
        method = "classical_fallback"

    plots = _make_plots(iq, mod, ai_decoded, classical)

    return jsonify({
        "bits_ai":        "".join(str(b) for b in ai_decoded[:128]),
        "bits_classical": "".join(str(b) for b in classical[:128]),
        "total_bits":     len(classical),
        "ber_ai":         None,
        "ber_classical":  None,
        "method":         method,
        "plots":          plots,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, host="0.0.0.0")

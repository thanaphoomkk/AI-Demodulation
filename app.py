"""
app.py  —  Flask Web App สำหรับ AI Demodulation
วิธีรัน:
    pip install flask numpy scipy matplotlib
    python app.py
แล้วเปิด http://localhost:5000
"""

from __future__ import annotations
import io, base64, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

from generate_signal import generate_signal, ALL_MODS, DIGITAL_MODS
from demod_classical import demodulate, iq_from_arrays

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# ------------------------------------------------------------------ helpers --

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _make_plots(iq: np.ndarray, mod: str, bits: list[int]) -> dict:
    """สร้างกราฟ 3 ใบ: waveform, constellation, bit stream"""

    # 1) Waveform
    n = min(300, len(iq))
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(iq.real[:n], label="I", lw=1.2)
    ax.plot(iq.imag[:n], label="Q", lw=1.2, alpha=0.75)
    ax.set_title(f"IQ Waveform — {mod.upper()}")
    ax.set_xlabel("Sample"); ax.legend(); ax.grid(alpha=0.3)
    wave_b64 = _fig_to_b64(fig)

    # 2) Constellation
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(iq.real, iq.imag, s=6, alpha=0.4)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"Constellation — {mod.upper()}")
    ax.set_xlabel("I"); ax.set_ylabel("Q")
    ax.axis("equal"); ax.grid(alpha=0.3)
    const_b64 = _fig_to_b64(fig)

    # 3) Bit stream (แสดง 64 บิตแรก)
    show = bits[:64]
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.step(range(len(show)), show, where="post", lw=1.5, color="steelblue")
    ax.set_ylim(-0.3, 1.3); ax.set_yticks([0, 1])
    ax.set_title(f"Bit Stream (64 บิตแรก) — {mod.upper()}")
    ax.set_xlabel("บิตที่"); ax.grid(alpha=0.3)
    bits_b64 = _fig_to_b64(fig)

    return {"waveform": wave_b64, "constellation": const_b64, "bitstream": bits_b64}


# ------------------------------------------------------------------ routes --

@app.route("/")
def index():
    return render_template("index.html", mods=ALL_MODS, digital_mods=DIGITAL_MODS)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """สร้างสัญญาณจำลองแล้วถอดรหัส"""
    data = request.get_json()
    mod    = data.get("mod", "qpsk").lower()
    snr    = float(data.get("snr", 10))
    n_sym  = int(data.get("n_symbols", 64))
    sps    = 8

    rng = np.random.default_rng()
    iq, src_bits = generate_signal(mod, n_symbols=n_sym, sps=sps,
                                   snr_db=snr, rng=rng)

    # ถอดรหัส (เฉพาะ digital)
    if mod in DIGITAL_MODS:
        decoded = demodulate(iq, mod, sps=sps)
        # BER เทียบกับบิตต้นทาง
        src = list(src_bits[:len(decoded)])
        errors = sum(a != b for a, b in zip(src, decoded))
        ber = errors / max(len(decoded), 1)
        bit_str = "".join(str(b) for b in decoded[:128])
    else:
        decoded = []
        ber = None
        bit_str = "(Analog — ไม่มีบิต)"

    plots = _make_plots(iq, mod, decoded)

    return jsonify({
        "bits": bit_str,
        "total_bits": len(decoded),
        "ber": round(ber, 4) if ber is not None else None,
        "plots": plots,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """รับไฟล์ CSV (I,Q ต่อบรรทัด) แล้วถอดรหัส"""
    mod = request.form.get("mod", "qpsk").lower()
    sps = int(request.form.get("sps", 1))
    f   = request.files.get("file")

    if not f:
        return jsonify({"error": "ไม่ได้แนบไฟล์"}), 400

    try:
        txt = f.read().decode("utf-8")
        rows = [line.strip().split(",") for line in txt.splitlines()
                if line.strip() and not line.startswith("#")]
        arr = np.array([[float(r[0]), float(r[1])] for r in rows if len(r) >= 2])
        iq = iq_from_arrays(arr[:, 0], arr[:, 1])
    except Exception as e:
        return jsonify({"error": f"อ่านไฟล์ไม่ได้: {e}"}), 400

    if mod not in DIGITAL_MODS:
        return jsonify({"error": f"อัปโหลดรองรับเฉพาะ {DIGITAL_MODS}"}), 400

    decoded = demodulate(iq, mod, sps=sps)
    bit_str = "".join(str(b) for b in decoded[:128])
    plots   = _make_plots(iq, mod, decoded)

    return jsonify({
        "bits": bit_str,
        "total_bits": len(decoded),
        "ber": None,
        "plots": plots,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")

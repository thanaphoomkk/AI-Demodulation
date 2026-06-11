"""
generate_signal.py
==================
สร้างสัญญาณจำลองสำหรับงาน AI Demodulation

รองรับ modulation 7 ประเภท:
    - BPSK   (digital, 1 บิต/สัญลักษณ์)
    - QPSK   (digital, 2 บิต/สัญลักษณ์)
    - QAM16  (digital, 4 บิต/สัญลักษณ์)
    - QAM64  (digital, 6 บิต/สัญลักษณ์)
    - QAM256 (digital, 8 บิต/สัญลักษณ์)
    - AM     (analog)
    - FM     (analog)

ฟังก์ชันหลักที่ใช้เทรนโมเดล:
    generate_dataset(...)  -> คืน X (IQ samples) และ y (บิต 0/1)

โครงสร้าง IQ:
    สัญญาณเชิงซ้อน complex baseband ถูกแยกเป็น 2 ช่อง [I, Q]
    X shape = (n_examples, T*sps, 2)
    y shape = (n_examples, T, bits_per_symbol)
"""

from __future__ import annotations
import numpy as np

# จำนวนบิตต่อสัญลักษณ์ของแต่ละ modulation แบบดิจิทัล
BITS_PER_SYMBOL = {
    "bpsk":  1,
    "qpsk":  2,
    "qam8":  3,
    "qam16": 4,
    "qam64": 6,
}

DIGITAL_MODS = list(BITS_PER_SYMBOL.keys())
ANALOG_MODS = ["am", "fm"]
ALL_MODS = DIGITAL_MODS + ANALOG_MODS


# --------------------------------------------------------------------------- #
#  ตัวสร้างบิตสุ่ม
# --------------------------------------------------------------------------- #
def random_bits(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, size=n).astype(np.int8)


# --------------------------------------------------------------------------- #
#  Mapper ดิจิทัล: บิต -> สัญลักษณ์เชิงซ้อน (normalize ให้พลังงานเฉลี่ย = 1)
# --------------------------------------------------------------------------- #
def bpsk_map(bits: np.ndarray) -> np.ndarray:
    """0 -> -1 , 1 -> +1"""
    return (2 * bits - 1).astype(np.complex64)


def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Gray-coded QPSK, 2 บิต/สัญลักษณ์"""
    b = bits.reshape(-1, 2)
    i = 2 * b[:, 0] - 1
    q = 2 * b[:, 1] - 1
    return ((i + 1j * q) / np.sqrt(2)).astype(np.complex64)


def qam16_map(bits: np.ndarray) -> np.ndarray:
    """16-QAM, 4 บิต/สัญลักษณ์ (2 บิตต่อแกน, Gray บนระดับ -3,-1,+1,+3)"""
    b = bits.reshape(-1, 4)
    # Gray mapping ของ 2 บิต -> ระดับ {-3,-1,+1,+3}
    gray = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    i = np.array([gray[(x, y)] for x, y in b[:, 0:2]])
    q = np.array([gray[(x, y)] for x, y in b[:, 2:4]])
    sym = (i + 1j * q).astype(np.complex64)
    # normalize พลังงานเฉลี่ยของ 16-QAM = 10
    return sym / np.sqrt(10)


def _gray_levels(n_bits: int) -> list[int]:
    """สร้างลำดับ Gray code -> ระดับแอมพลิจูด สำหรับ n_bits บิต (n_bits ต้องเป็นเลขคู่)"""
    n = 1 << n_bits          # จำนวนระดับ = 2^n_bits
    levels = list(range(-(n - 1), n, 2))   # -7,-5,...,+7  หรือ -15,...,+15
    # สร้าง Gray order: xor ลำดับ -> index ของ levels
    gray_order = [i ^ (i >> 1) for i in range(n)]
    return [levels[gray_order[i]] for i in range(n)]


def _qam_map(bits: np.ndarray, bps: int) -> np.ndarray:
    """Generic square QAM mapper (bps ต้องหารด้วย 2 ลงตัว)"""
    bits_per_axis = bps // 2
    b = bits.reshape(-1, bps)
    lut = _gray_levels(bits_per_axis)
    norm = np.sqrt(np.mean(np.array(lut) ** 2) * 2)

    def bits_to_level(row_bits):
        idx = int("".join(str(x) for x in row_bits), 2)
        return lut[idx]

    i_vals = np.array([bits_to_level(row[:bits_per_axis]) for row in b])
    q_vals = np.array([bits_to_level(row[bits_per_axis:]) for row in b])
    return ((i_vals + 1j * q_vals) / norm).astype(np.complex64)


def qam8_map(bits: np.ndarray) -> np.ndarray:
    """Cross-8QAM, 3 บิต/สัญลักษณ์ — ใช้ตารางเดียวกับ demod_classical"""
    from demod_classical import QAM8_CONST, QAM8_BITS
    # สร้าง LUT: bin(b2b1b0) -> symbol
    lut = np.zeros(8, dtype=np.complex64)
    for sym_idx, brow in enumerate(QAM8_BITS):
        bin_val = brow[0]*4 + brow[1]*2 + brow[2]
        lut[bin_val] = QAM8_CONST[sym_idx]
    b = bits.reshape(-1, 3)
    idx = (b[:, 0]*4 + b[:, 1]*2 + b[:, 2]).astype(int)
    return lut[idx]


def qam64_map(bits: np.ndarray) -> np.ndarray:
    return _qam_map(bits, 6)


DIGITAL_MAPPERS = {
    "bpsk":  bpsk_map,
    "qpsk":  qpsk_map,
    "qam8":  qam8_map,
    "qam16": qam16_map,
    "qam64": qam64_map,
}


# --------------------------------------------------------------------------- #
#  Pulse shaping (Root-Raised-Cosine) + upsample
# --------------------------------------------------------------------------- #
def rrc_filter(beta: float, sps: int, span: int = 6) -> np.ndarray:
    """สร้าง impulse response ของ Root-Raised-Cosine filter"""
    n = np.arange(-span * sps / 2, span * sps / 2 + 1)
    t = n / sps
    h = np.zeros_like(t, dtype=np.float64)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return (h / np.sqrt(np.sum(h ** 2))).astype(np.float32)


def upsample_and_shape(symbols: np.ndarray, sps: int, beta: float = 0.35) -> np.ndarray:
    """แทรกศูนย์ระหว่างสัญลักษณ์ (upsample) แล้วผ่าน RRC filter"""
    up = np.zeros(len(symbols) * sps, dtype=np.complex64)
    up[::sps] = symbols
    h = rrc_filter(beta, sps)
    shaped = np.convolve(up, h, mode="same")
    return shaped.astype(np.complex64)


# --------------------------------------------------------------------------- #
#  Analog modulation: AM / FM
# --------------------------------------------------------------------------- #
def _message_signal(n: int, rng: np.random.Generator, f: float = 0.02) -> np.ndarray:
    """สัญญาณข้อความ (เสียง) จำลอง = ผลรวมไซน์หลายความถี่"""
    t = np.arange(n)
    phase = rng.uniform(0, 2 * np.pi)
    msg = (
        np.sin(2 * np.pi * f * t + phase)
        + 0.5 * np.sin(2 * np.pi * 2 * f * t)
        + 0.3 * np.sin(2 * np.pi * 3.5 * f * t)
    )
    return (msg / np.max(np.abs(msg))).astype(np.float32)


def am_modulate(n: int, rng: np.random.Generator, mod_index: float = 0.7) -> np.ndarray:
    """AM (DSB) -> complex baseband"""
    msg = _message_signal(n, rng)
    env = 1.0 + mod_index * msg
    return env.astype(np.complex64)  # envelope จริง, Q = 0


def fm_modulate(n: int, rng: np.random.Generator, kf: float = 0.3) -> np.ndarray:
    """FM -> complex baseband exp(j * 2*pi*kf * integral(msg))"""
    msg = _message_signal(n, rng)
    phase = 2 * np.pi * kf * np.cumsum(msg)
    return np.exp(1j * phase).astype(np.complex64)


# --------------------------------------------------------------------------- #
#  เพิ่ม noise (AWGN) ตามค่า SNR
# --------------------------------------------------------------------------- #
def add_awgn(signal: np.ndarray, snr_db: float,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """เติม Additive White Gaussian Noise ให้ได้ SNR ตามต้องการ (dB)"""
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return (signal + noise).astype(np.complex64)


# --------------------------------------------------------------------------- #
#  สร้างสัญญาณ 1 ชุด (ใช้สำหรับ visualization / test)
# --------------------------------------------------------------------------- #
def generate_signal(mod_type: str, n_symbols: int = 128, sps: int = 8,
                    snr_db: float = 15.0,
                    rng: np.random.Generator | None = None):
    """
    คืน (iq, bits)
        iq   : ndarray complex baseband (มี noise แล้ว)
        bits : ndarray บิตต้นทาง (เฉพาะ digital), หรือ None สำหรับ analog
    """
    rng = rng or np.random.default_rng()
    mod_type = mod_type.lower()

    if mod_type in DIGITAL_MODS:
        bps = BITS_PER_SYMBOL[mod_type]
        bits = random_bits(n_symbols * bps, rng)
        symbols = DIGITAL_MAPPERS[mod_type](bits)
        clean = upsample_and_shape(symbols, sps)
        iq = add_awgn(clean, snr_db, rng)
        return iq, bits

    elif mod_type == "am":
        clean = am_modulate(n_symbols * sps, rng)
        return add_awgn(clean, snr_db, rng), None

    elif mod_type == "fm":
        clean = fm_modulate(n_symbols * sps, rng)
        return add_awgn(clean, snr_db, rng), None

    raise ValueError(f"ไม่รู้จัก modulation: {mod_type} (เลือกจาก {ALL_MODS})")


# --------------------------------------------------------------------------- #
#  สร้าง dataset สำหรับเทรน/ทดสอบโมเดล (เฉพาะ digital modulation)
# --------------------------------------------------------------------------- #
def generate_dataset(mod_type: str = "qpsk",
                     n_examples: int = 2000,
                     symbols_per_example: int = 64,
                     sps: int = 8,
                     snr_range=(0, 20),
                     seed: int | None = None):
    """
    สร้างชุดข้อมูลสำหรับ demodulator

    คืน:
        X : (n_examples, symbols_per_example*sps, 2)  ช่อง [I, Q]
        y : (n_examples, symbols_per_example, bits_per_symbol) บิต 0/1
    """
    mod_type = mod_type.lower()
    if mod_type not in DIGITAL_MODS:
        raise ValueError(
            f"การเทรน demodulator รองรับเฉพาะ digital mods {DIGITAL_MODS}"
        )

    rng = np.random.default_rng(seed)
    bps = BITS_PER_SYMBOL[mod_type]
    T = symbols_per_example

    X = np.zeros((n_examples, T * sps, 2), dtype=np.float32)
    y = np.zeros((n_examples, T, bps), dtype=np.float32)

    for k in range(n_examples):
        # สุ่ม SNR ในแต่ละตัวอย่าง เพื่อให้โมเดลทนต่อ noise หลายระดับ
        snr_db = rng.uniform(*snr_range)
        bits = random_bits(T * bps, rng)
        symbols = DIGITAL_MAPPERS[mod_type](bits)
        clean = upsample_and_shape(symbols, sps)
        iq = add_awgn(clean, snr_db, rng)

        X[k, :, 0] = iq.real
        X[k, :, 1] = iq.imag
        y[k] = bits.reshape(T, bps)

    return X, y


if __name__ == "__main__":
    # ทดสอบเร็ว ๆ ว่าสร้างสัญญาณได้ครบทุกชนิด
    print("ตัวอย่างการสร้างสัญญาณแต่ละชนิด:")
    for m in ALL_MODS:
        iq, bits = generate_signal(m, n_symbols=32, snr_db=10)
        nbits = "-" if bits is None else len(bits)
        print(f"  {m:6s} -> IQ length={len(iq):4d}, bits={nbits}")

    X, y = generate_dataset("qpsk", n_examples=10)
    print(f"\nDataset QPSK: X={X.shape}, y={y.shape}")

"""
demod_classical.py
==================
ถอดรหัส IQ samples กลับเป็นบิต 0/1 แบบ Classical (ไม่ใช้ AI)
ใช้แค่ numpy ล้วนๆ
"""

from __future__ import annotations
import numpy as np


def _nearest_symbol(iq_point: complex, constellation: np.ndarray) -> int:
    """หา index ของสัญลักษณ์ที่ใกล้ที่สุด (minimum distance) ด้วย numpy"""
    return int(np.argmin(np.abs(constellation - iq_point)))


# ---------- BPSK ----------
BPSK_CONST = np.array([-1+0j, 1+0j], dtype=np.complex128)
BPSK_BITS  = [[0], [1]]

# ---------- QPSK (Gray) ----------
QPSK_CONST = np.array([
    (-1-1j) / np.sqrt(2),
    (-1+1j) / np.sqrt(2),
    ( 1-1j) / np.sqrt(2),
    ( 1+1j) / np.sqrt(2),
], dtype=np.complex128)
QPSK_BITS = [[0, 0], [0, 1], [1, 0], [1, 1]]

def _build_square_qam(bits_per_axis: int):
    """สร้าง constellation + bit table สำหรับ Square QAM ทุกขนาด"""
    n = 1 << bits_per_axis          # จำนวนระดับต่อแกน
    levels = list(range(-(n - 1), n, 2))   # เช่น -3,-1,+1,+3 สำหรับ n=4
    # Gray code สำหรับ n ระดับ
    gray_bits = [[int(b) for b in format(i ^ (i >> 1), f'0{bits_per_axis}b')]
                 for i in range(n)]
    norm = np.sqrt(np.mean(np.array(levels) ** 2) * 2)
    const, bit_table = [], []
    for gi, lv_i in zip(gray_bits, levels):
        for gq, lv_q in zip(gray_bits, levels):
            const.append(complex(lv_i, lv_q) / norm)
            bit_table.append(gi + gq)
    return np.array(const, dtype=np.complex128), bit_table


# ---------- 16-QAM (Gray) ----------
QAM16_CONST, QAM16_BITS   = _build_square_qam(2)   # 2 บิต/แกน -> 4 บิต/symbol
# ---------- 64-QAM (Gray) ----------
QAM64_CONST, QAM64_BITS   = _build_square_qam(3)   # 3 บิต/แกน -> 6 บิต/symbol
# ---------- 256-QAM (Gray) ----------
QAM256_CONST, QAM256_BITS = _build_square_qam(4)   # 4 บิต/แกน -> 8 บิต/symbol

CONSTELLATIONS = {
    "bpsk":   (BPSK_CONST,   BPSK_BITS),
    "qpsk":   (QPSK_CONST,   QPSK_BITS),
    "qam16":  (QAM16_CONST,  QAM16_BITS),
    "qam64":  (QAM64_CONST,  QAM64_BITS),
    "qam256": (QAM256_CONST, QAM256_BITS),
}


def downsample(iq: np.ndarray, sps: int) -> np.ndarray:
    """เก็บ sample กลาง (peak) ของแต่ละสัญลักษณ์"""
    offset = sps // 2
    return iq[offset::sps]


def demodulate(iq: np.ndarray, mod_type: str, sps: int = 1) -> list[int]:
    """
    รับ IQ array (complex) คืนบิต 0/1

    iq       : complex ndarray  (ถ้า sps>1 จะ downsample ก่อน)
    mod_type : 'bpsk' | 'qpsk' | 'qam16' | 'qam64' | 'qam256'
    sps      : samples ต่อสัญลักษณ์
    """
    mod_type = mod_type.lower()
    if mod_type not in CONSTELLATIONS:
        raise ValueError(f"รองรับเฉพาะ {list(CONSTELLATIONS.keys())}")

    const, bit_table = CONSTELLATIONS[mod_type]

    # downsample ถ้าจำเป็น
    symbols = downsample(iq, sps) if sps > 1 else iq

    bits: list[int] = []
    for sym in symbols:
        idx = _nearest_symbol(sym, const)
        bits.extend(bit_table[idx])
    return bits


def iq_from_arrays(i_arr: np.ndarray, q_arr: np.ndarray) -> np.ndarray:
    """รวม I, Q array เป็น complex array"""
    return (np.asarray(i_arr, dtype=np.float64)
            + 1j * np.asarray(q_arr, dtype=np.float64))

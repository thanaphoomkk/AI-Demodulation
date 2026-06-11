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


# ---------- 8-QAM (Cross pattern) ----------
# ตาราง hardcoded: symbol index -> (I, Q, [b2, b1, b0])
# จุด 8 จุด ถูก assign บิตแบบ Gray-like เพื่อให้เพื่อนบ้านต่างกัน 1 บิต
_QAM8_TABLE = [
    # I,    Q,    bits
    (-1.0, -1.0, [0, 0, 0]),
    (-1.0, +1.0, [0, 0, 1]),
    (+1.0, -1.0, [0, 1, 0]),
    (+1.0, +1.0, [0, 1, 1]),
    (-3.0, +0.0, [1, 0, 0]),
    (+3.0, +0.0, [1, 0, 1]),
    (+0.0, -3.0, [1, 1, 0]),
    (+0.0, +3.0, [1, 1, 1]),
]

def _build_cross_qam8():
    raw = np.array([complex(r[0], r[1]) for r in _QAM8_TABLE], dtype=np.complex128)
    norm = np.sqrt(np.mean(np.abs(raw) ** 2))
    return raw / norm, [r[2] for r in _QAM8_TABLE]

QAM8_CONST, QAM8_BITS = _build_cross_qam8()
# ---------- 16-QAM (Gray) ----------
QAM16_CONST, QAM16_BITS  = _build_square_qam(2)
# ---------- 64-QAM (Gray) ----------
QAM64_CONST, QAM64_BITS  = _build_square_qam(3)
CONSTELLATIONS = {
    "bpsk":  (BPSK_CONST,  BPSK_BITS),
    "qpsk":  (QPSK_CONST,  QPSK_BITS),
    "qam8":  (QAM8_CONST,  QAM8_BITS),
    "qam16": (QAM16_CONST, QAM16_BITS),
    "qam64": (QAM64_CONST, QAM64_BITS),
}


def downsample(iq: np.ndarray, sps: int) -> np.ndarray:
    """เก็บ sample กลาง (peak) ของแต่ละสัญลักษณ์"""
    offset = sps // 2
    return iq[offset::sps]


def matched_filter(iq: np.ndarray, sps: int, beta: float = 0.35) -> np.ndarray:
    """
    Matched Filter สำหรับ RRC pulse shaping

    หลักการ: เมื่อ TX ใช้ RRC filter ปั้นพัลส์ RX ต้องผ่าน RRC filter อีกครั้ง
    (RRC * RRC = RC) เพื่อกำจัด ISI และ maximize SNR ที่จุด sampling

    ขั้นตอน:
      1. convolve สัญญาณ noisy กับ RRC filter เดียวกับ TX
      2. downsample ที่กลาง symbol (ซึ่งตอนนี้ไม่มี ISI แล้ว)
    """
    if sps <= 1:
        return iq
    from generate_signal import rrc_filter
    h = rrc_filter(beta, sps).astype(np.complex64)
    # TX ใช้ RRC ไปแล้ว 1 ครั้ง (delay=half_len) RX filter เพิ่มอีก 1 ครั้ง
    # รวม total group delay = 2 * half_len  แต่ mode="full" เริ่มต้นที่ index 0
    # ดังนั้น sample แรกที่ไม่มี ISI อยู่ที่ index = 2*half_len + sps//2
    half_len = (len(h) - 1) // 2
    filtered = np.convolve(iq, h, mode="full")
    # TX ใช้ RRC แล้ว 1 รอบ  RX filter เพิ่มอีก 1 รอบ
    # peak ของ RC pulse (RRC*RRC) เกิดที่ delay = half_len samples หลัง iq เริ่ม
    # แต่ mode="full" ทำให้ output ยาวขึ้น → peak อยู่ที่ half_len - sps//2 + 1
    start = half_len - sps // 2 + 1
    return filtered[start::sps]


def demodulate(iq: np.ndarray, mod_type: str, sps: int = 1) -> list[int]:
    """
    รับ IQ array (complex) คืนบิต 0/1
    ถ้า sps > 1 จะผ่าน Matched Filter (integrate-and-dump) ก่อนตัดสินใจ

    iq       : complex ndarray
    mod_type : 'bpsk' | 'qpsk' | 'qam8' | 'qam16' | 'qam64'
    sps      : samples ต่อสัญลักษณ์
    """
    mod_type = mod_type.lower()
    if mod_type not in CONSTELLATIONS:
        raise ValueError(f"รองรับเฉพาะ {list(CONSTELLATIONS.keys())}")

    const, bit_table = CONSTELLATIONS[mod_type]

    # Matched Filter (integrate-and-dump) แทน downsample เฉยๆ
    symbols = matched_filter(iq, sps) if sps > 1 else iq

    bits: list[int] = []
    for sym in symbols:
        idx = _nearest_symbol(sym, const)
        bits.extend(bit_table[idx])
    return bits


def iq_from_arrays(i_arr: np.ndarray, q_arr: np.ndarray) -> np.ndarray:
    """รวม I, Q array เป็น complex array"""
    return (np.asarray(i_arr, dtype=np.float64)
            + 1j * np.asarray(q_arr, dtype=np.float64))

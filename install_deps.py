"""
install_deps.py
ติดตั้ง library ที่จำเป็นทั้งหมดแบบอัตโนมัติ

วิธีใช้:
    python install_deps.py

สคริปต์นี้จะเรียก pip ติดตั้งทุกแพ็กเกจใน requirements.txt
หากเครื่องไม่มี GPU ก็ยังรันได้ (TensorFlow จะใช้ CPU)
"""

import subprocess
import sys
import importlib

# ชื่อแพ็กเกจ (สำหรับ pip)  ->  ชื่อโมดูลที่ใช้ import (สำหรับเช็ก)
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "tensorflow": "tensorflow",
}


def is_installed(module_name: str) -> bool:
    """ตรวจว่าโมดูลถูกติดตั้งแล้วหรือยัง"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def pip_install(package: str) -> None:
    """ติดตั้งแพ็กเกจด้วย pip"""
    print(f"  -> กำลังติดตั้ง {package} ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", package]
    )


def main() -> None:
    print("=" * 60)
    print(" ตรวจสอบและติดตั้ง library สำหรับ AI Demodulation")
    print("=" * 60)

    # อัปเกรด pip ก่อน เพื่อลดปัญหาตอนติดตั้ง tensorflow
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
        )
    except subprocess.CalledProcessError:
        print("  (ข้ามการอัปเกรด pip)")

    for pip_name, module_name in PACKAGES.items():
        if is_installed(module_name):
            print(f"[OK]   {pip_name} ติดตั้งแล้ว")
        else:
            print(f"[MISS] {pip_name} ยังไม่ได้ติดตั้ง")
            pip_install(pip_name)

    print("=" * 60)
    print(" ติดตั้งครบทุกแพ็กเกจเรียบร้อย ✔")
    print("=" * 60)


if __name__ == "__main__":
    main()

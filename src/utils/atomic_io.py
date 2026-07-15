"""Write CSV atomik -- tulis ke file temporer di direktori yang sama lalu
`os.replace()` ke path tujuan (atomik di POSIX maupun Windows selama masih di
volume yang sama). Dipakai untuk file tracking yang ditulis ulang PENUH tiap
kali ada baris baru (predictions_log.csv/accuracy_log.csv sudah 5-6MB+) --
tanpa ini, proses yang mati di tengah `to_csv()` (mis. dibunuh paksa lewat
Task Scheduler timeout) bisa meninggalkan file setengah-tertulis, menghapus
SELURUH riwayat alih-alih cuma kehilangan satu baris terbaru. Lihat audit
codebase 2026-07-12."""

import os
import tempfile

import pandas as pd


def atomic_write_csv(df: pd.DataFrame, path: str, **to_csv_kwargs) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_atomic_", suffix=".csv")
    try:
        os.close(fd)
        df.to_csv(tmp_path, **to_csv_kwargs)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

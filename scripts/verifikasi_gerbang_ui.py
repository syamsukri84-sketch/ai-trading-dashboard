"""Gerbang verifikasi UI: AppTest matriks 4 halaman x 3 mode (12 kombinasi).

Jalankan dari root proyek dengan venv proyek:
    .venv\\Scripts\\python.exe scripts\\verifikasi_gerbang_ui.py

Lulus = 12 kombinasi tanpa exception. Perkiraan durasi: 5-20 menit
(tiap kombinasi mengeksekusi dashboard penuh tanpa browser).
Exit code 0 = LULUS, 1 = GAGAL (jangan commit sebelum lulus).
"""

import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")  # emoji aman di console Windows
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)  # streamlit_app mengimpor run_analysis dkk dari root proyek

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["🏠 Keputusan", "🔬 Riset & Akurasi", "⚙️ Operasional", "📰 Sentimen Pasar"]
MODES = ["Pemula", "Trader", "Audit"]


def main() -> int:
    failures = []
    total_start = time.time()
    for page in PAGES:
        for mode in MODES:
            label = f"[{page} | {mode}]"
            start = time.time()
            try:
                at = AppTest.from_file("streamlit_app.py", default_timeout=180)
                at.run()
                for r in at.sidebar.radio:  # pilih radio berdasarkan isi opsinya
                    opts = [str(o) for o in (r.options or [])]
                    if page in opts:
                        r.set_value(page)
                    elif mode in opts:
                        r.set_value(mode)
                at.run()
                n_exc = len(at.exception)
                dur = time.time() - start
                if n_exc == 0:
                    print(f"OK    {label} {dur:.1f}s")
                else:
                    msgs = [str(e.value)[:200] for e in at.exception]
                    failures.append((label, msgs))
                    print(f"GAGAL {label} {dur:.1f}s -> {msgs}")
            except Exception as err:  # kegagalan harness itu sendiri juga = gagal
                failures.append((label, [str(err)[:200]]))
                print(f"ERROR {label} -> {err}")
    print("-" * 60)
    print(f"Total durasi: {(time.time() - total_start) / 60:.1f} menit")
    if failures:
        print(f"HASIL: GAGAL ({len(failures)}/12 kombinasi bermasalah) -- JANGAN COMMIT.")
        for label, msgs in failures:
            print(" ", label, msgs)
        return 1
    print("HASIL: LULUS -- 12/12 kombinasi, 0 exception. Aman untuk commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

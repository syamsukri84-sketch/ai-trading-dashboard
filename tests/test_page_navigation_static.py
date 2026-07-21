"""Penjaga statis restrukturisasi navigasi 4 halaman (2026-07-21).

Latar: pada st.tabs SEMUA blok dieksekusi tiap rerun sehingga blok bebas
berbagi variabel. Setelah pindah ke navigasi kondisional, dependensi
lintas-blok yang tidak sehalaman = NameError saat runtime. Test ini
membuktikan lewat AST (tanpa perlu Streamlit) bahwa:
1. Keenam blok tab lama kini berada di bawah guard `if _show_*` yang benar.
2. Setiap nama yang dipakai satu blok tetapi didefinisikan blok lain berada
   pada guard/halaman yang SAMA dan bloknya dieksekusi lebih dulu.
3. File tetap valid secara sintaks.
"""

import ast
import os

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
TABS = ["tab_beranda", "tab_daily", "tab_update", "tab_ranking", "tab_accuracy", "tab_sentiment"]
EXPECTED_GUARD = {
    "tab_daily": "_show_keputusan",
    "tab_beranda": "_show_keputusan",
    "tab_update": "_show_operasional",
    "tab_ranking": "_show_riset",
    "tab_accuracy": "_show_riset",
    "tab_sentiment": "_show_sentimen",
}


def _parse():
    with open(APP_PATH, encoding="utf-8-sig") as f:
        return ast.parse(f.read())


def _find_guarded_blocks(tree):
    """{tab: (guard_name, with_node, lineno)} untuk pola `if _show_x:` -> `with tab_y:`."""
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name):
            continue
        guard = node.test.id
        if not guard.startswith("_show_"):
            continue
        for child in node.body:
            if isinstance(child, ast.With) and len(child.items) == 1:
                ctx = child.items[0].context_expr
                if isinstance(ctx, ast.Name) and ctx.id in TABS:
                    found[ctx.id] = (guard, child, node.lineno)
    return found


def test_semua_blok_tab_terjaga_guard_yang_benar():
    blocks = _find_guarded_blocks(_parse())
    assert set(blocks) == set(TABS), f"Blok tab tanpa guard: {set(TABS) - set(blocks)}"
    for tab, (guard, _n, _ln) in blocks.items():
        assert guard == EXPECTED_GUARD[tab], f"{tab} dijaga {guard}, seharusnya {EXPECTED_GUARD[tab]}"


def test_dependensi_lintas_blok_sehalaman_dan_berurutan():
    import builtins

    tree = _parse()
    blocks = _find_guarded_blocks(tree)
    names = {}
    for tab, (guard, node, lineno) in blocks.items():
        assigned, used = set(), set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                (assigned if isinstance(n.ctx, ast.Store) else used).add(n.id)
            elif isinstance(n, ast.FunctionDef):
                assigned.add(n.name)
            elif isinstance(n, ast.alias):
                assigned.add((n.asname or n.name).split(".")[0])
        names[tab] = {"assigned": assigned, "used": used, "guard": guard, "lineno": lineno}
    # nama yang tersedia global (di luar keenam blok) dianggap aman
    guarded_nodes = {id(node) for _t, (_g, node, _l) in blocks.items()}
    global_names = set(dir(builtins))
    for node in tree.body:
        inner = node.body if isinstance(node, ast.If) else [node]
        for child in inner:
            if id(child) in guarded_nodes:
                continue
            for n in ast.walk(child):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    global_names.add(n.id)
                elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    global_names.add(n.name)
                elif isinstance(n, ast.alias):
                    global_names.add((n.asname or n.name).split(".")[0])
    problems = []
    for tab, info in names.items():
        external = info["used"] - info["assigned"] - global_names - set(TABS)
        for name in sorted(external):
            providers = [t for t in names if t != tab and name in names[t]["assigned"]]
            if not providers:
                continue  # bukan dependensi antar blok (mis. variabel exception lokal)
            ok = any(
                names[p]["guard"] == info["guard"] and names[p]["lineno"] < info["lineno"]
                for p in providers
            )
            if not ok:
                problems.append(f"{tab} memakai '{name}' dari {providers} yang beda halaman/urutan")
    assert not problems, "Dependensi lintas halaman terdeteksi:\n" + "\n".join(problems)


def test_file_valid_dan_navigasi_terdefinisi():
    tree = _parse()  # gagal parse = gagal test
    src_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for required in ["NAV_PAGES", "_nav_page", "_show_keputusan", "_show_riset", "_show_operasional", "_show_sentimen"]:
        assert required in src_names, f"Simbol navigasi hilang: {required}"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Profile viewer:
- List and filter *_profile.npy files (pattern: <beta>_<hash>_profile.npy)
- Overlay multiple profiles (convert to log2(||b*_i||))
- Optional measured profile overlay with scaling
- Interactive REPL with history (↑/↓)
- CN11 overlay using Simulator.CN11
- Save/List/Load/Delete CN11 configs (only CN11)
- Per-hash scaling for loaded profiles (subtract log2(scale) for matching hash)
"""

import os
import re
import atexit
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# ---------- Optional: Simulator for CN11 ----------
try:
    from estimator import *  # expects: Simulator.CN11(d, n_eff, q, beta, xi=..., tau=..., dual=False)
except Exception:
    Simulator = None

# ---------- REPL command history ----------
def enable_history():
    try:
        import readline
        histfile = os.path.expanduser("~/.profile_viewer_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file, histfile)
        readline.set_history_length(1000)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

# ---------- Filenames ----------
FNAME_RE = re.compile(r'(?P<beta>\d+)_(?P<hash>[0-9a-fA-F]+)_profile\.npy$')

# ---------- CN11 config store ----------
CN11_STORE_PATH = os.path.expanduser("~/.profile_viewer_cn11_configs.json")

def load_cn11_db() -> Dict[str, dict]:
    try:
        with open(CN11_STORE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_cn11_db(db: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(CN11_STORE_PATH), exist_ok=True)
    with open(CN11_STORE_PATH, "w") as f:
        json.dump(db, f, indent=2)

# ---------- Scan & group ----------
def scan_profiles(dirpath: str) -> List[Dict]:
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Directory not found: {dirpath}")
    out = []
    for fn in os.listdir(dirpath):
        m = FNAME_RE.match(fn)
        if not m:
            continue
        beta = int(m.group('beta'))
        h = m.group('hash').lower()
        path = os.path.join(dirpath, fn)
        out.append({'path': path, 'beta': beta, 'hash': h, 'name': fn})
    out.sort(key=lambda x: (x['hash'], x['beta'], x['name']))
    return out

def group_by_hash(profiles: List[Dict]) -> Dict[str, List[Dict]]:
    g = defaultdict(list)
    for p in profiles:
        g[p['hash']].append(p)
    for h in g:
        g[h].sort(key=lambda x: (x['beta'], x['name']))
    return dict(g)

# ---------- Format conversions ----------
def to_log2_norm(arr: np.ndarray, form: str) -> np.ndarray:
    """
    Convert various formats to r = log2(||b*_i||), i.e., non-squared log2 norm.
    form:
      - 'log2'        : already log2(||b||)
      - 'log2_norm2'  : log2(||b||^2) -> divide by 2
      - 'log10_norm2' : log10(||b||^2) -> convert to log2 then /2
      - 'norm'        : ||b|| -> apply log2
      - 'norm2'       : ||b||^2 -> apply log2 then /2
    """
    form = form.lower()
    if form == 'log2':
        return arr.astype(float)
    elif form == 'log2_norm2':
        return (arr.astype(float)) / 2.0
    elif form == 'log10_norm2':
        LOG10_2 = np.log10(2.0)
        return arr.astype(float) / (2.0 * LOG10_2)
    elif form == 'norm':
        return np.log2(np.maximum(arr.astype(float), np.finfo(float).tiny))
    elif form == 'norm2':
        return 0.5 * np.log2(np.maximum(arr.astype(float), np.finfo(float).tiny))
    else:
        raise ValueError(f"Unknown form: {form}")

def load_saved_profile(path: str, saved_form: str = 'log2_norm2', scale: float = 1.0) -> np.ndarray:
    """
    Load a *_profile.npy and convert to log2(||b||).
    Then subtract log2(scale) if scale != 1.0.
    """
    raw = np.load(path)
    r = to_log2_norm(raw, saved_form)
    if scale and scale != 1.0:
        r = r - np.log2(float(scale))
    return r

def load_measured(path: str,
                  measured_form: str = 'log2',
                  scaling_y: float = 1.0) -> np.ndarray:
    raw = np.load(path)
    r = to_log2_norm(raw, measured_form)
    if scaling_y and scaling_y != 1.0:
        r = r - np.log2(float(scaling_y))
    return r

# ---------- Selection & table ----------
def print_table(profiles: List[Dict], title: str = "Profiles") -> None:
    if not profiles:
        print("(empty)")
        return
    width = max(len(p['name']) for p in profiles)
    print(f"\n=== {title} ({len(profiles)}) ===")
    print("ID  β    Hash                          Filename")
    print("--  ---  ----------------------------  " + "-" * width)
    for i, p in enumerate(profiles, 1):
        print(f"{i:>2}  {p['beta']:<3}  {p['hash']:<28}  {p['name']}")
    print("")

def filter_profiles(profiles: List[Dict],
                    betas: Optional[List[int]] = None,
                    hashes: Optional[List[str]] = None) -> List[Dict]:
    sel = profiles
    if betas:
        bset = set(betas)
        sel = [p for p in sel if p['beta'] in bset]
    if hashes:
        hset = set(h.lower() for h in hashes)
        sel = [p for p in sel if p['hash'] in hset]
    return sel

def parse_id_list(s: str) -> List[int]:
    """
    Parse "1,2,5-8" -> [1,2,5,6,7,8]
    """
    out = []
    for chunk in s.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            a, b = int(a), int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(chunk))
    return out

# ---------- CN11 helpers ----------
def compute_cn11_curve(d: int, n: int, zeta: int, q: int, beta: int,
                       xi: float, tau: float, dual: bool = False, max_loops : int = 8) -> np.ndarray:
    if Simulator is None:
        raise RuntimeError("Module 'Simulator' not found: cannot compute CN11.")
    n_eff = n - zeta
    r = Simulator.CN11(d, n_eff, q, beta, xi=xi, tau=tau, dual=dual, max_loops=max_loops)  # squared norms
    r = np.asarray(r, dtype=float)
    r = 0.5 * np.log2(np.maximum(r, np.finfo(float).tiny))  # -> log2(||b*||)
    return r

# ---------- Plot ----------
def plot_profiles(selection: List[Dict],
                  saved_form: str = 'log2_norm2',
                  hash_scale: Optional[Dict[str, float]] = None,
                  measured: Optional[np.ndarray] = None,
                  title: Optional[str] = None,
                  save_path: Optional[str] = None,
                  xrange: Optional[Tuple[int, int]] = None,
                  # CN11 overlay:
                  cn11_enabled: bool = False,
                  cn11_d: Optional[int] = None,
                  cn11_n: Optional[int] = None,
                  cn11_zeta: Optional[int] = None,
                  cn11_q: Optional[int] = None,
                  cn11_betas: Optional[List[int]] = None,
                  cn11_use_selection_betas: bool = True,
                  cn11_xi: float = 1.0,
                  cn11_tau: float = 0.0,
                  cn11_dual: bool = False,
                  cn11_max_loops: int = 8) -> None:

    hash_scale = hash_scale or {}

    curves = []
    if measured is not None:
        curves.append(('measured', measured))

    for p in selection:
        s = float(hash_scale.get(p['hash'], 1.0))
        r = load_saved_profile(p['path'], saved_form=saved_form, scale=s)
        lbl = f"β={p['beta']} | {p['hash'][:8]}"
        curves.append((lbl, r))

    if not curves and not cn11_enabled:
        print("Nothing to plot.")
        return

    d_min_existing = min(len(r) for _, r in curves) if curves else None

    cn11_curves = []
    if cn11_enabled:
        if cn11_n is None or cn11_zeta is None or cn11_q is None:
            print("[CN11] Missing parameters: n, zeta and q are required.")
        else:
            betas_auto = sorted({p['beta'] for p in selection}) if (selection and cn11_use_selection_betas) else []
            betas_final = []
            if cn11_betas:
                betas_final.extend(list(cn11_betas))
            betas_final.extend(betas_auto)
            betas_final = sorted({int(b) for b in betas_final})
            if not betas_final:
                print("[CN11] No β specified (nor found from selection).")
            else:
                d_for_cn = cn11_d if (cn11_d is not None) else (d_min_existing or 0)
                if d_for_cn <= 0:
                    print("[CN11] Invalid dimension d.")
                else:
                    for b in betas_final:
                        try:
                            rcn = compute_cn11_curve(d_for_cn, cn11_n, cn11_zeta, cn11_q,
                                                     b, cn11_xi, cn11_tau, cn11_dual, cn11_max_loops)
                            cn11_curves.append((f"CN11 β={b}", rcn))
                        except Exception as e:
                            print(f"[CN11] Failed for β={b}: {e}")

    all_curves = curves + cn11_curves
    if not all_curves:
        print("Nothing to plot.")
        return

    d_min = min(len(r) for _, r in all_curves)
    all_curves = [(lbl, r[:d_min]) for lbl, r in all_curves]

    lo, hi = 1, d_min
    if xrange:
        lo = max(1, int(xrange[0]))
        hi = min(d_min, int(xrange[1]))
        if lo > hi:
            lo, hi = hi, lo
    idx = np.arange(lo, hi + 1)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for lbl, r in all_curves:
        ax.plot(idx, r[lo-1:hi], lw=1.6, label=lbl)

    ax.set_xlabel("index i")
    ax.set_ylabel("log2 ||b*_i||")
    ttl = title or "Basis profile superposition"
    ax.set_title(ttl)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    ref_name, ref = all_curves[0]
    print(f"\nΔ stats vs '{ref_name}':")
    for lbl, r in all_curves[1:]:
        diff = r - ref
        print(f"  {lbl:>18}: mean={diff.mean():.4f}, std={diff.std(ddof=1):.4f}, max|Δ|={np.abs(diff).max():.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

# ---------- REPL (interactive) ----------
def interactive_shell(all_profiles: List[Dict],
                      saved_form: str,
                      hash_scale: Dict[str, float],
                      measured: Optional[np.ndarray],
                      title: Optional[str],
                      save_path: Optional[str],
                      xrange: Optional[Tuple[int, int]]) -> None:
    enable_history()
    selection: List[Dict] = []

    cn11_cfg = dict(
        enabled=False, d=None, n=None, zeta=None, q=None,
        betas=set(), use_selection_betas=True,
        xi=1.0, tau=0.0, dual=False, max_loops=8
    )

    print("Interactive mode. Commands:")
    print("  list                              -> list all profiles")
    print("  add <ids>                         -> add by IDs (e.g., add 1,3-5)")
    print("  add hash <hex>                    -> add all with a given hash")
    print("  add beta <b>                      -> add all with β=b")
    print("  rm <ids>                          -> remove IDs from current selection")
    print("  clear                             -> clear selection")
    print("  show                              -> show current selection")
    print("  plot                              -> plot selection (+ measured/CN11 if enabled)")
    print("  scale set <hash> <s>              -> set per-hash scale (subtract log2(s))")
    print("  scale show                        -> show current per-hash scales")
    print("  scale clear <hash|all>            -> clear one/all scales")
    print("  cn11 on|off                       -> enable/disable CN11 overlay")
    print("  cn11 set d|n|zeta|q|xi|tau|maxloops <val>  -> set CN11 parameters")
    print("  cn11 set dual 0|1                 -> set 'dual' flag")
    print("  cn11 add beta <b>                 -> add β to CN11 list")
    print("  cn11 clear beta                   -> clear explicit CN11 β list")
    print("  cn11 use sel on|off               -> use selection betas for CN11")
    print("  cn11 show                         -> show CN11 config")
    print("  cn11 save <name>                  -> save current CN11 config")
    print("  cn11 list                         -> list saved CN11 configs")
    print("  cn11 load <name>                  -> load a saved CN11 config")
    print("  cn11 del <name>                   -> delete a saved CN11 config")
    print("  quit                              -> exit\n")

    def cn11_show():
        cfg = cn11_cfg
        print("\n[CN11 config]")
        print(f"  enabled={cfg['enabled']}, d={cfg['d']}, n={cfg['n']}, zeta={cfg['zeta']}, q={cfg['q']}")
        print(f"  xi={cfg['xi']}, tau={cfg['tau']}, dual={cfg['dual']}, max_loops={cfg['max_loops']}")
        print(f"  use_selection_betas={cfg['use_selection_betas']}, betas_explicit={sorted(cfg['betas'])}\n")

    def scales_show():
        if not hash_scale:
            print("(no per-hash scales)")
            return
        print("\n[Per-hash scales]  (subtract log2(scale) when hash matches)")
        for h, s in sorted(hash_scale.items()):
            print(f"  {h}: {s}")
        print("")

    while True:
        try:
            cmd = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not cmd:
            continue

        if cmd == 'list':
            print_table(all_profiles, "All profiles")
        elif cmd.startswith('add hash '):
            h = cmd.split(None, 2)[2].strip().lower()
            picked = [p for p in all_profiles if p['hash'] == h]
            new = [p for p in picked if p not in selection]
            selection.extend(new)
            print(f"Added {len(new)} from hash {h}.")
        elif cmd.startswith('add beta '):
            try:
                b = int(cmd.split(None, 2)[2])
            except Exception:
                print("Usage: add beta <int>")
                continue
            picked = [p for p in all_profiles if p['beta'] == b]
            new = [p for p in picked if p not in selection]
            selection.extend(new)
            print(f"Added {len(new)} from β={b}.")
        elif cmd.startswith('add '):
            ids = parse_id_list(cmd[4:].strip())
            ok = []
            for i in ids:
                if 1 <= i <= len(all_profiles):
                    p = all_profiles[i-1]
                    if p not in selection:
                        selection.append(p)
                        ok.append(i)
            print(f"Added IDs: {ok}" if ok else "Nothing added.")
        elif cmd.startswith('rm '):
            ids = set(parse_id_list(cmd[3:].strip()))
            keep = []
            removed = 0
            for idx, p in enumerate(selection, 1):
                if idx in ids:
                    removed += 1
                else:
                    keep.append(p)
            selection = keep
            print(f"Removed {removed} item(s).")
        elif cmd == 'clear':
            selection = []
            print("Selection cleared.")
        elif cmd == 'show':
            print_table(selection, "Current selection")
        elif cmd == 'plot':
            plot_profiles(selection, saved_form=saved_form,
                          hash_scale=hash_scale,
                          measured=measured, title=title,
                          save_path=save_path, xrange=xrange,
                          cn11_enabled=cn11_cfg['enabled'],
                          cn11_d=cn11_cfg['d'],
                          cn11_n=cn11_cfg['n'],
                          cn11_zeta=cn11_cfg['zeta'],
                          cn11_q=cn11_cfg['q'],
                          cn11_betas=sorted(cn11_cfg['betas']) if cn11_cfg['betas'] else None,
                          cn11_use_selection_betas=cn11_cfg['use_selection_betas'],
                          cn11_xi=cn11_cfg['xi'],
                          cn11_tau=cn11_cfg['tau'],
                          cn11_dual=cn11_cfg['dual'],
                          cn11_max_loops=cn11_cfg['max_loops'])
        elif cmd.startswith('scale '):
            parts = cmd.split()
            if len(parts) >= 2 and parts[1] == 'show':
                scales_show()
            elif len(parts) == 4 and parts[1] == 'set':
                h = parts[2].lower()
                try:
                    s = float(parts[3])
                except Exception:
                    print("Usage: scale set <hash> <scale>")
                    continue
                hash_scale[h] = s
                print(f"Scale set for {h} = {s} (subtract log2({s})).")
            elif len(parts) == 3 and parts[1] == 'clear':
                target = parts[2].lower()
                if target == 'all':
                    hash_scale.clear()
                    print("All per-hash scales cleared.")
                else:
                    if target in hash_scale:
                        del hash_scale[target]
                        print(f"Scale cleared for {target}.")
                    else:
                        print("Hash not found in scale map.")
            else:
                print("scale commands: 'scale show' | 'scale set <hash> <scale>' | 'scale clear <hash|all>'")
        elif cmd.startswith('cn11 '):
            sub = cmd[5:].strip()
            if sub == 'on':
                cn11_cfg['enabled'] = True
                print("CN11 overlay enabled.")
            elif sub == 'off':
                cn11_cfg['enabled'] = False
                print("CN11 overlay disabled.")
            elif sub.startswith('set '):
                parts = sub.split()
                if len(parts) < 3:
                    print("Usage: cn11 set <key> <value>")
                    continue
                key, val = parts[1], " ".join(parts[2:])
                if key in ('d', 'n', 'zeta', 'q'):
                    cn11_cfg[key] = int(eval(val)) # for 2**k
                elif key in ('xi', 'tau'):
                    cn11_cfg[key] = float(val)
                elif key == 'dual':
                    cn11_cfg['dual'] = (val.strip().lower() in ('1', 'true', 'on', 'yes'))
                elif key == 'maxloops':
                    cn11_cfg['max_loops'] = int(val)
                else:
                    print("Keys: d|n|zeta|q|xi|tau|dual|maxloops")
                cn11_show()
            elif sub.startswith('add beta '):
                try:
                    b = int(sub.split(None, 2)[2])
                    cn11_cfg['betas'].add(b)
                    print(f"CN11: added β={b}")
                except Exception:
                    print("Usage: cn11 add beta <int>")
            elif sub == 'clear beta':
                cn11_cfg['betas'].clear()
                print("CN11: explicit β list cleared.")
            elif sub.startswith('use sel '):
                flag = sub.split(None, 2)[2].strip().lower()
                cn11_cfg['use_selection_betas'] = (flag in ('on', '1', 'true', 'yes'))
                print(f"CN11: use_selection_betas={cn11_cfg['use_selection_betas']}")
            elif sub == 'show':
                cn11_show()
            elif sub.startswith('save '):
                name = sub.split(None, 1)[1].strip()
                if not name:
                    print("Usage: cn11 save <name>")
                else:
                    db = load_cn11_db()
                    # serialize set to list
                    payload = dict(cn11_cfg)
                    payload['betas'] = sorted(payload['betas'])
                    db[name] = payload
                    save_cn11_db(db)
                    print(f"[CN11] Saved config '{name}'.")
            elif sub == 'list':
                db = load_cn11_db()
                if not db:
                    print("(no saved CN11 configs)")
                else:
                    print("\n[Saved CN11 configs]")
                    for k in sorted(db.keys()):
                        cfg = db[k]
                        print(f"- {k}: d={cfg.get('d')}, n={cfg.get('n')}, zeta={cfg.get('zeta')}, q={cfg.get('q')}, "
                              f"xi={cfg.get('xi')}, tau={cfg.get('tau')}, dual={cfg.get('dual')}, maxloops={cfg.get('max_loops')},"
                              f"use_sel={cfg.get('use_selection_betas')}, betas={cfg.get('betas')}")
                    print("")
            elif sub.startswith('load '):
                name = sub.split(None, 1)[1].strip()
                db = load_cn11_db()
                if name not in db:
                    print(f"[CN11] No config named '{name}'.")
                else:
                    cfg = db[name]
                    cn11_cfg.update(cfg)
                    # turn list back into set
                    cn11_cfg['betas'] = set(cfg.get('betas', []))
                    cn11_cfg['enabled'] = True
                    print(f"[CN11] Loaded config '{name}' and enabled overlay.")
                    cn11_show()
            elif sub.startswith('del '):
                name = sub.split(None, 1)[1].strip()
                db = load_cn11_db()
                if name in db:
                    del db[name]
                    save_cn11_db(db)
                    print(f"[CN11] Deleted config '{name}'.")
                else:
                    print(f"[CN11] No config named '{name}'.")
            else:
                print("Unknown 'cn11' command.")
        elif cmd in ('quit', 'exit'):
            print("Bye.")
            break
        else:
            print("Unknown command.")

# ---------- CLI parsing helpers ----------
def parse_hash_scale_args(items: Optional[List[str]]) -> Dict[str, float]:
    """
    Parse repeated '--hash-scale HASH=SCALE' into a dict {hash: scale}.
    """
    mapping: Dict[str, float] = {}
    if not items:
        return mapping
    for it in items:
        if '=' not in it:
            raise ValueError("Bad --hash-scale format. Use HASH=SCALE and repeat as needed.")
        h, s = it.split('=', 1)
        h = h.strip().lower()
        s = float(s.strip())
        mapping[h] = s
    return mapping

# ---------- Main / CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Viewer for *_profile.npy with filters, REPL, per-hash scaling, and CN11 overlay."
    )
    ap.add_argument("--dir", default="saved_basis",
                    help="Directory containing *_profile.npy (default: saved_basis)")
    ap.add_argument("--list", action="store_true",
                    help="List all profiles and exit")
    ap.add_argument("--beta", type=int, nargs="*", default=None,
                    help="Filter by beta (multiple allowed)")
    ap.add_argument("--hash", type=str, nargs="*", default=None,
                    help="Filter by hash (multiple allowed)")
    ap.add_argument("--ids", type=str, default=None,
                    help="Select by IDs after listing (e.g., '1,3-5')")
    ap.add_argument("--interactive", action="store_true",
                    help="Interactive REPL")
    # Formats
    ap.add_argument("--saved-form", default=None,
                    choices=["log2", "log2_norm2", "log10_norm2", "norm", "norm2"],
                    help="Format of *_profile.npy. assume 'log2'; ")
    # Per-hash scaling
    ap.add_argument("--hash-scale", action="append", default=None,
                    help="Per-hash scale: HASH=SCALE (subtract log2(SCALE)); repeatable.")
    # Measured
    ap.add_argument("--measured-path", type=str, default=None,
                    help="Path to measured profile (.npy)")
    ap.add_argument("--measured-form", default="log2",
                    choices=["log2", "log2_norm2", "log10_norm2", "norm", "norm2"],
                    help="Format of measured profile (default: log2)")
    ap.add_argument("--scaling-y", type=float, default=1.0,
                    help="Subtract log2(scaling_y) from measured profile")
    # Viz
    ap.add_argument("--title", type=str, default=None,
                    help="Plot title")
    ap.add_argument("--save", type=str, default=None,
                    help="Path to save the figure (if omitted: plt.show())")
    ap.add_argument("--xrange", type=str, default=None,
                    help="Index range to plot, e.g., '10:200'")
    # CN11 CLI
    ap.add_argument("--cn11", action="store_true",
                    help="Enable CN11 overlay")
    ap.add_argument("--cn11-d", type=int, default=None,
                    help="CN11 dimension d (default: min length of plotted curves)")
    ap.add_argument("--cn11-n", type=int, default=None,
                    help="CN11 parameter n (required if --cn11)")
    ap.add_argument("--cn11-zeta", type=int, default=None,
                    help="CN11 parameter zeta (required if --cn11)")
    ap.add_argument("--cn11-q", type=int, default=None,
                    help="CN11 modulus q (required if --cn11)")
    ap.add_argument("--cn11-betas", type=int, nargs="*", default=None,
                    help="Explicit β list for CN11 (default: use selection's betas)")
    ap.add_argument("--cn11-no-selection-betas", action="store_true",
                    help="Do not use selection betas for CN11")
    ap.add_argument("--cn11-xi", type=float, default=1.0,
                    help="CN11 xi parameter (default: 1.0)")
    ap.add_argument("--cn11-tau", type=float, default=0.0,
                    help="CN11 tau parameter (default: 0.0)")
    ap.add_argument("--cn11-dual", action="store_true",
                    help="Enable CN11 dual mode")
    ap.add_argument("--cn11-max_loops", type=int, default=8,
                    help="Number of loops allowed in the simulator")
    # CN11 config storage (save/list/load/delete)
    ap.add_argument("--cn11-config-save", type=str, default=None,
                    help="Save current CN11 config under a name")
    ap.add_argument("--cn11-configs-list", action="store_true",
                    help="List saved CN11 configs and exit")
    ap.add_argument("--cn11-config-load", type=str, default=None,
                    help="Load a saved CN11 config by name")
    ap.add_argument("--cn11-config-delete", type=str, default=None,
                    help="Delete a saved CN11 config by name and exit")

    args = ap.parse_args()

    # Decide default saved_form based on directory name if not provided
    if args.saved_form is None:
        base = os.path.basename(os.path.abspath(args.dir))
        args.saved_form = 'log2'

    # Handle CN11 config store ops that exit immediately
    if args.cn11_configs_list:
        db = load_cn11_db()
        if not db:
            print("(no saved CN11 configs)")
        else:
            print("\n[Saved CN11 configs]")
            for k in sorted(db.keys()):
                cfg = db[k]
                print(f"- {k}: d={cfg.get('d')}, n={cfg.get('n')}, zeta={cfg.get('zeta')}, q={cfg.get('q')}, "
                      f"xi={cfg.get('xi')}, tau={cfg.get('tau')}, dual={cfg.get('dual')}, maxloops={cfg.get('max_loops')},"
                      f"use_sel={cfg.get('use_selection_betas')}, betas={cfg.get('betas')}")
            print("")
        return

    if args.cn11_config_delete:
        db = load_cn11_db()
        name = args.cn11_config_delete
        if name in db:
            del db[name]
            save_cn11_db(db)
            print(f"Deleted CN11 config '{name}'.")
        else:
            print(f"No CN11 config named '{name}'.")
        return

    # Load profiles
    profiles = scan_profiles(args.dir)

    if args.list and not args.interactive and args.ids is None:
        print_table(profiles, "All profiles")
        return

    # Hash scaling map
    hash_scale_map = parse_hash_scale_args(args.hash_scale)

    # Filters
    filtered = filter_profiles(profiles, betas=args.beta, hashes=args.hash)

    # Range
    xr = None
    if args.xrange:
        try:
            a, b = args.xrange.split(':', 1)
            xr = (int(a), int(b))
        except Exception:
            raise ValueError("Bad --xrange format. Use 'start:end' (e.g., 1:200).")

    # Measured
    measured = None
    if args.measured_path:
        measured = load_measured(args.measured_path, args.measured_form, args.scaling_y)

    # Interactive REPL
    if args.interactive:
        print_table(profiles, "All profiles")
        interactive_shell(profiles, saved_form=args.saved_form,
                          hash_scale=hash_scale_map,
                          measured=measured, title=args.title,
                          save_path=args.save, xrange=xr)
        return

    # Selection via IDs (apply on filtered list if present)
    selection = filtered
    if args.ids:
        base_list = filtered if filtered else profiles
        print_table(base_list, "Filtered profiles" if filtered else "All profiles")
        ids = parse_id_list(args.ids)
        picked = []
        for i in ids:
            if 1 <= i <= len(base_list):
                picked.append(base_list[i-1])
        selection = picked

    # CN11 config load (merges + turns overlay on)
    if args.cn11_config_load:
        db = load_cn11_db()
        name = args.cn11_config_load
        if name not in db:
            print(f"No CN11 config named '{name}'.")
            return
        cfg = db[name]
        # Fill in CN11 args from loaded config
        args.cn11 = True
        args.cn11_d = cfg.get('d', args.cn11_d)
        args.cn11_n = cfg.get('n', args.cn11_n)
        args.cn11_zeta = cfg.get('zeta', args.cn11_zeta)
        args.cn11_q = cfg.get('q', args.cn11_q)
        args.cn11_betas = cfg.get('betas', args.cn11_betas)
        args.cn11_xi = cfg.get('xi', args.cn11_xi)
        args.cn11_tau = cfg.get('tau', args.cn11_tau)
        args.cn11_dual = cfg.get('dual', args.cn11_dual)
        args.cn11_dual = cfg.get('max_loops', args.max_loops)
        if cfg.get('use_selection_betas') is not None and cfg.get('use_selection_betas') is False:
            args.cn11_no_selection_betas = True
        print(f"[CN11] Loaded config '{name}'.")

    # CN11 config save (does not exit; it saves current CN11 flags)
    if args.cn11_config_save:
        payload = {
            "d": args.cn11_d,
            "n": args.cn11_n,
            "zeta": args.cn11_zeta,
            "q": args.cn11_q,
            "betas": args.cn11_betas if args.cn11_betas else [],
            "use_selection_betas": (not args.cn11_no_selection_betas),
            "xi": args.cn11_xi,
            "tau": args.cn11_tau,
            "dual": args.cn11_dual,
            "max_loops": args.cn11_max_loops
        }
        db = load_cn11_db()
        db[args.cn11_config_save] = payload
        save_cn11_db(db)
        print(f"[CN11] Saved config '{args.cn11_config_save}'.")

    if not selection and measured is None and not args.cn11:
        print("Nothing selected. Use --beta/--hash/--ids or --interactive.")
        return

    # Plot
    plot_profiles(
        selection,
        saved_form=args.saved_form,
        hash_scale=hash_scale_map,
        measured=measured, title=args.title, save_path=args.save, xrange=xr,
        cn11_enabled=args.cn11,
        cn11_d=args.cn11_d,
        cn11_n=args.cn11_n,
        cn11_zeta=args.cn11_zeta,
        cn11_q=args.cn11_q,
        cn11_betas=args.cn11_betas,
        cn11_use_selection_betas=(not args.cn11_no_selection_betas),
        cn11_xi=args.cn11_xi,
        cn11_tau=args.cn11_tau,
        cn11_dual=args.cn11_dual,
        cn11_max_loops=args.cn11_max_loops,
    )

if __name__ == "__main__":
    main()
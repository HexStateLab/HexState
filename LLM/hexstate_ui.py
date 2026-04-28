#!/usr/bin/env python3
"""
HexState Quantizer — Desktop UI
================================
Dark-themed tkinter GUI for the full HexState quantization pipeline.

Features:
  1. Compile C library (libhexstate_q2k.so)
  2. Convert HuggingFace → GGUF (via convert_hf_to_gguf.py)
  3. Generate importance matrix (via generate_imatrix.py)
  4. Quantize GGUF → Q2_K/Q4_0 hybrid (via hexstate_requantize.py)
  5. Validate output GGUF (via validate_gguf.py)

Usage:
    python3 hexstate_ui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import threading
import os
import sys
import time
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
# Color Palette — HexState Dark Theme
# ═══════════════════════════════════════════════════════════════════════════
BG           = "#0d1117"
BG_CARD      = "#161b22"
BG_INPUT     = "#0d1117"
BG_HOVER     = "#1c2333"
FG           = "#c9d1d9"
FG_DIM       = "#8b949e"
FG_BRIGHT    = "#e6edf3"
ACCENT       = "#58a6ff"
ACCENT_HOVER = "#79c0ff"
GREEN        = "#3fb950"
ORANGE       = "#d29922"
RED          = "#f85149"
PURPLE       = "#bc8cff"
BORDER       = "#30363d"
CONSOLE_BG   = "#010409"
CONSOLE_FG   = "#b1bac4"


class HexStateUI:
    def __init__(self, root):
        self.root = root
        self.root.title("⬡ HexState Quantizer")
        self.root.configure(bg=BG)
        self.root.geometry("960x820")
        self.root.minsize(800, 700)
        self.running_process = None

        self._setup_styles()
        self._build_ui()

    # ── Styles ──────────────────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=BG, foreground=FG, fieldbackground=BG_INPUT,
                         bordercolor=BORDER, troughcolor=BG)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("TLabel", background=BG, foreground=FG, font=("Inter", 10))
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG)
        style.configure("Header.TLabel", background=BG, foreground=FG_BRIGHT,
                         font=("Inter", 18, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=FG_DIM,
                         font=("Inter", 9))
        style.configure("Section.TLabel", background=BG_CARD, foreground=ACCENT,
                         font=("Inter", 11, "bold"))
        style.configure("Status.TLabel", background=BG, foreground=GREEN,
                         font=("Inter", 9, "bold"))
        style.configure("TEntry", fieldbackground=BG_INPUT, foreground=FG,
                         insertcolor=FG, borderwidth=1)

        # Buttons
        style.configure("Action.TButton", background=ACCENT, foreground="#ffffff",
                         font=("Inter", 10, "bold"), padding=(16, 8),
                         borderwidth=0)
        style.map("Action.TButton",
                  background=[("active", ACCENT_HOVER), ("disabled", BORDER)],
                  foreground=[("disabled", FG_DIM)])
        style.configure("Browse.TButton", background=BG_CARD, foreground=FG_DIM,
                         font=("Inter", 9), padding=(8, 4))
        style.map("Browse.TButton", background=[("active", BG_HOVER)])
        style.configure("Stop.TButton", background=RED, foreground="#ffffff",
                         font=("Inter", 10, "bold"), padding=(16, 8))
        style.map("Stop.TButton", background=[("active", "#da3633")])

        # Notebook
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_CARD, foreground=FG_DIM,
                         padding=(14, 6), font=("Inter", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", BG), ("active", BG_HOVER)],
                  foreground=[("selected", ACCENT)])

    # ── Main Layout ─────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = ttk.Frame(self.root)
        hdr.pack(fill="x", padx=20, pady=(16, 4))
        ttk.Label(hdr, text="⬡  HexState Quantizer", style="Header.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(hdr, textvariable=self.status_var, style="Status.TLabel").pack(side="right")

        ttk.Label(self.root, text="D₆ Hadamard Error Shaping  ·  Z₆ Belief Propagation  ·  Triality-Weighted Scoring",
                  style="Sub.TLabel").pack(anchor="w", padx=22)

        # Separator
        sep = tk.Frame(self.root, height=1, bg=BORDER)
        sep.pack(fill="x", padx=20, pady=(12, 0))

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(8, 0))

        self._build_compile_tab()
        self._build_convert_tab()
        self._build_imatrix_tab()
        self._build_quantize_tab()
        self._build_validate_tab()

        # Console output
        console_frame = ttk.Frame(self.root)
        console_frame.pack(fill="both", expand=True, padx=20, pady=(8, 12))

        cf_header = ttk.Frame(console_frame)
        cf_header.pack(fill="x")
        ttk.Label(cf_header, text="Console Output", style="Section.TLabel").pack(side="left")
        self.stop_btn = ttk.Button(cf_header, text="⏹ Stop", style="Stop.TButton",
                                    command=self._stop_process, state="disabled")
        self.stop_btn.pack(side="right")
        clear_btn = ttk.Button(cf_header, text="Clear", style="Browse.TButton",
                                command=self._clear_console)
        clear_btn.pack(side="right", padx=(0, 6))

        self.console = scrolledtext.ScrolledText(
            console_frame, bg=CONSOLE_BG, fg=CONSOLE_FG,
            insertbackground=CONSOLE_FG, font=("JetBrains Mono", 9),
            height=12, wrap="word", borderwidth=1, relief="solid",
            highlightbackground=BORDER, highlightthickness=1)
        self.console.pack(fill="both", expand=True, pady=(6, 0))
        self.console.tag_configure("info", foreground=ACCENT)
        self.console.tag_configure("ok", foreground=GREEN)
        self.console.tag_configure("warn", foreground=ORANGE)
        self.console.tag_configure("err", foreground=RED)
        self.console.configure(state="disabled")

        self._log("HexState Quantizer UI initialized.\n", "info")

    # ── Tab Builders ────────────────────────────────────────────────────
    def _make_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.pack(fill="both", expand=True, padx=4, pady=6)
        return card

    def _file_row(self, parent, label_text, var, filetypes=None, is_dir=False, is_save=False):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text=label_text, style="Card.TLabel", width=16, anchor="e").pack(side="left")
        entry = ttk.Entry(row, textvariable=var, width=50)
        entry.pack(side="left", padx=(8, 4), fill="x", expand=True)

        def browse():
            if is_dir:
                path = filedialog.askdirectory(initialdir=SCRIPT_DIR)
            elif is_save:
                path = filedialog.asksaveasfilename(
                    initialdir=SCRIPT_DIR, filetypes=filetypes or [("GGUF", "*.gguf")])
            else:
                path = filedialog.askopenfilename(
                    initialdir=SCRIPT_DIR, filetypes=filetypes or [("All", "*.*")])
            if path:
                var.set(path)

        ttk.Button(row, text="Browse…", style="Browse.TButton", command=browse).pack(side="left")
        return row

    # ── 1. Compile ──────────────────────────────────────────────────────
    def _build_compile_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ⚙ Compile  ")
        card = self._make_card(tab)
        ttk.Label(card, text="Compile HPC Engine", style="Section.TLabel").pack(anchor="w")
        ttk.Label(card, text="Builds libhexstate_q2k.so from hexstate_quantize.c\n"
                  "Requires: gcc, libgmp-dev, libmpfr-dev, libomp-dev",
                  style="Card.TLabel", justify="left").pack(anchor="w", pady=(4, 12))
        ttk.Button(card, text="🔨  Compile Library", style="Action.TButton",
                   command=self._run_compile).pack(anchor="w")

    # ── 2. Convert HF ──────────────────────────────────────────────────
    def _build_convert_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📦 Convert HF  ")
        card = self._make_card(tab)
        ttk.Label(card, text="Convert HuggingFace → GGUF", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.hf_dir_var = tk.StringVar()
        self.hf_out_var = tk.StringVar()
        self.hf_type_var = tk.StringVar(value="bf16")
        self._file_row(card, "HF Model Dir:", self.hf_dir_var, is_dir=True)
        self._file_row(card, "Output GGUF:", self.hf_out_var, is_save=True,
                       filetypes=[("GGUF", "*.gguf")])
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 12))
        ttk.Label(row, text="Output Type:", style="Card.TLabel", width=16, anchor="e").pack(side="left")
        for val in ["bf16", "f16", "f32"]:
            ttk.Radiobutton(row, text=val.upper(), variable=self.hf_type_var,
                            value=val).pack(side="left", padx=4)
        ttk.Button(card, text="📦  Convert to GGUF", style="Action.TButton",
                   command=self._run_convert).pack(anchor="w")

    # ── 3. iMatrix ──────────────────────────────────────────────────────
    def _build_imatrix_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  📊 iMatrix  ")
        card = self._make_card(tab)
        ttk.Label(card, text="Generate Importance Matrix", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.imat_model_var = tk.StringVar()
        self.imat_cal_var = tk.StringVar(value=os.path.join(SCRIPT_DIR, "calibration_data.txt"))
        self.imat_out_var = tk.StringVar(value=os.path.join(SCRIPT_DIR, "imatrix.dat"))
        self.imat_chunks_var = tk.StringVar(value="100")
        self._file_row(card, "Source GGUF:", self.imat_model_var,
                       filetypes=[("GGUF", "*.gguf")])
        self._file_row(card, "Calibration Text:", self.imat_cal_var,
                       filetypes=[("Text", "*.txt")])
        self._file_row(card, "Output iMatrix:", self.imat_out_var, is_save=True,
                       filetypes=[("DAT", "*.dat"), ("GGUF", "*.gguf")])
        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 12))
        ttk.Label(row, text="Chunks:", style="Card.TLabel", width=16, anchor="e").pack(side="left")
        ttk.Entry(row, textvariable=self.imat_chunks_var, width=8).pack(side="left", padx=8)
        ttk.Button(card, text="📊  Generate iMatrix", style="Action.TButton",
                   command=self._run_imatrix).pack(anchor="w")

    # ── 4. Quantize ─────────────────────────────────────────────────────
    def _build_quantize_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ⬡ Quantize  ")
        card = self._make_card(tab)
        ttk.Label(card, text="Quantize GGUF → Q2_K / Q4_0 Hybrid", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.q_input_var = tk.StringVar()
        self.q_output_var = tk.StringVar()
        self.q_imat_var = tk.StringVar()
        self._file_row(card, "Input GGUF:", self.q_input_var,
                       filetypes=[("GGUF", "*.gguf")])
        self._file_row(card, "Output GGUF:", self.q_output_var, is_save=True,
                       filetypes=[("GGUF", "*.gguf")])
        self._file_row(card, "iMatrix (opt.):", self.q_imat_var,
                       filetypes=[("DAT/GGUF", "*.dat *.gguf")])
        ttk.Button(card, text="⬡  Quantize with HPC Engine", style="Action.TButton",
                   command=self._run_quantize).pack(anchor="w")

    # ── 5. Validate ─────────────────────────────────────────────────────
    def _build_validate_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  ✓ Validate  ")
        card = self._make_card(tab)
        ttk.Label(card, text="Validate GGUF Output", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.v_input_var = tk.StringVar()
        self._file_row(card, "GGUF File:", self.v_input_var,
                       filetypes=[("GGUF", "*.gguf")])
        ttk.Button(card, text="✓  Validate GGUF", style="Action.TButton",
                   command=self._run_validate).pack(anchor="w")

    # ── Console Helpers ─────────────────────────────────────────────────
    def _log(self, text, tag=None):
        self.console.configure(state="normal")
        self.console.insert("end", text, tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _set_status(self, text, color=GREEN):
        self.status_var.set(text)
        style = ttk.Style()
        style.configure("Status.TLabel", foreground=color)

    # ── Process Runner ──────────────────────────────────────────────────
    def _run_command(self, cmd, label=""):
        if self.running_process is not None:
            self._log("⚠ A process is already running!\n", "warn")
            return

        self._set_status(f"Running: {label}…", ORANGE)
        self.stop_btn.configure(state="normal")
        self._log(f"\n{'═'*60}\n  {label}\n  $ {' '.join(cmd)}\n{'═'*60}\n\n", "info")

        def worker():
            start = time.time()
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=SCRIPT_DIR, text=True, bufsize=1)
                self.running_process = proc
                for line in iter(proc.stdout.readline, ""):
                    # Strip \r progress lines for cleaner display
                    clean = line.rstrip()
                    if clean:
                        tag = None
                        if "RMSE=" in clean:
                            tag = "ok"
                        elif "WARNING" in clean or "⚠" in clean:
                            tag = "warn"
                        elif "Error" in clean or "error" in clean or "FAIL" in clean:
                            tag = "err"
                        self.root.after(0, self._log, clean + "\n", tag)
                proc.wait()
                elapsed = time.time() - start
                code = proc.returncode
                if code == 0:
                    self.root.after(0, self._log,
                                   f"\n✓ Completed in {elapsed:.1f}s\n", "ok")
                    self.root.after(0, self._set_status, "Done ✓", GREEN)
                elif code == -15 or code == -9:
                    self.root.after(0, self._log, "\n⏹ Stopped by user\n", "warn")
                    self.root.after(0, self._set_status, "Stopped", ORANGE)
                else:
                    self.root.after(0, self._log,
                                   f"\n✗ Failed (exit code {code})\n", "err")
                    self.root.after(0, self._set_status, f"Error (code {code})", RED)
            except Exception as e:
                self.root.after(0, self._log, f"\n✗ Exception: {e}\n", "err")
                self.root.after(0, self._set_status, "Error", RED)
            finally:
                self.running_process = None
                self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_process(self):
        if self.running_process:
            self.running_process.terminate()
            self._log("\n⏹ Sending SIGTERM…\n", "warn")

    # ── Command Builders ────────────────────────────────────────────────
    def _run_compile(self):
        self._run_command(["make", "-f", "makefile.quantize"], "Compile HPC Engine")

    def _run_convert(self):
        hf_dir = self.hf_dir_var.get().strip()
        out = self.hf_out_var.get().strip()
        if not hf_dir:
            self._log("⚠ Please select a HuggingFace model directory.\n", "warn")
            return
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "convert_hf_to_gguf.py"), hf_dir]
        if out:
            cmd += ["--outfile", out]
        cmd += ["--outtype", self.hf_type_var.get()]
        self._run_command(cmd, "Convert HF → GGUF")

    def _run_imatrix(self):
        model = self.imat_model_var.get().strip()
        cal = self.imat_cal_var.get().strip()
        out = self.imat_out_var.get().strip()
        if not model or not cal:
            self._log("⚠ Please select model GGUF and calibration text.\n", "warn")
            return
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "generate_imatrix.py"),
               model, cal, "-o", out, "--chunks", self.imat_chunks_var.get()]
        self._run_command(cmd, "Generate iMatrix")

    def _run_quantize(self):
        inp = self.q_input_var.get().strip()
        out = self.q_output_var.get().strip()
        if not inp or not out:
            self._log("⚠ Please select input and output GGUF files.\n", "warn")
            return
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "hexstate_requantize.py"), inp, out]
        imat = self.q_imat_var.get().strip()
        if imat:
            cmd += ["--imatrix", imat]
        self._run_command(cmd, "Quantize GGUF (HPC Engine)")

    def _run_validate(self):
        inp = self.v_input_var.get().strip()
        if not inp:
            self._log("⚠ Please select a GGUF file to validate.\n", "warn")
            return
        script = os.path.join(SCRIPT_DIR, "validate_gguf.py")
        if not os.path.exists(script):
            self._log("⚠ validate_gguf.py not found.\n", "warn")
            return
        self._run_command([sys.executable, script, inp], "Validate GGUF")


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass
    app = HexStateUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

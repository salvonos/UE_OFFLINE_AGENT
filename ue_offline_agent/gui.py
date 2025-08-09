# gui.py
import os
import threading
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import requests
import json

from main import (
    ensure_packages,
    load_config,
    LocalChatLLM,
    CONFIG_PATH,
)

from settings_language import (
    AVAILABLE_LANGUAGES,
    set_language,
    T,
    get_system_prompt,
)

from gui_Style import apply_theme

# Cataloghi modelli per lingua (multilingual per default)
MODEL_CATALOGS = {
    "en": [
        {
            "name": "Llama 3.2 3B Instruct Q4_K_M (~2.0 GB)",
            "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true",
            "filename": "llama-3.2-3b-instruct-q4_k_m.gguf",
            "ctx_train": 131072,
        },
        {
            "name": "Llama 3.1 8B Instruct Q4_K_M (~4.6 GB)",
            "url": "https://huggingface.co/bartowski/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true",
            "filename": "llama-3.1-8b-instruct-q4_k_m.gguf",
            "ctx_train": 131072,
        },
        {
            "name": "Mistral 7B Instruct v0.3 Q4_K_M (~4.1 GB)",
            "url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true",
            "filename": "mistral-7b-instruct-v0.3-q4_k_m.gguf",
            "ctx_train": 32768,
        },
    ],
    "it": [],
    "es": [],
    "fr": [],
    "de": [],
}

def models_for(lang: str):
    return MODEL_CATALOGS[lang] if MODEL_CATALOGS.get(lang) else MODEL_CATALOGS["en"]

MAX_HISTORY_TURNS = 24

class ChatGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_packages()
        self.cfg = load_config()

        # Lingua e tema
        self.lang = self.cfg.get("language", "it")
        self.theme = self.cfg.get("theme", "dark")
        set_language(self.lang)

        self.title(T("app_title"))
        self.geometry("900x600")

        self._ensure_model_folder()
        self._build_menu()
        self._build_ui()
        self._bind_shortcuts()

        # Applica tema (default dark)
        self._apply_theme(self.theme)

        self.generating = False
        self.current_reply_parts = []

        self._refresh_system_prompt_in_cfg()
        self.history = [{"role": "system", "content": self.cfg.get("system_prompt", get_system_prompt())}]

        self._init_llm()
        self._append_info(T("hint_ready"))

    # ---------- Lingua ----------
    def _apply_language_to_gui(self):
        self.title(T("app_title"))
        self._rebuild_menu()
        self.send_btn.config(text=T("btn_send"))
        self.stop_btn.config(text=T("btn_stop"))
        self.status_var.set(T("status_ready"))

    def _refresh_system_prompt_in_cfg(self):
        self.cfg["language"] = self.lang
        self.cfg["system_prompt"] = get_system_prompt()
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=2)
        except Exception:
            pass

    # ---------- Tema ----------
    def _apply_theme(self, theme: str):
        self.theme = theme
        apply_theme(self, theme, self.chat_text, self.entry, CONFIG_PATH, self.cfg)

    # ---------- Folders ----------
    def _ensure_model_folder(self):
        models_dir = self.cfg.get("models_dir")
        if not models_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            self.cfg["models_dir"] = models_dir
        else:
            os.makedirs(models_dir, exist_ok=True)

    # ---------- Menu ----------
    def _build_menu(self):
        self.menubar = tk.Menu(self)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label=T("menu_save"), command=self._save_chat)
        self.file_menu.add_separator()
        self.file_menu.add_command(label=T("menu_exit"), command=self.destroy)
        self.menubar.add_cascade(label=T("menu_file"), menu=self.file_menu)

        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.edit_menu.add_command(label=T("menu_settings"), command=self._open_settings)
        self.menubar.add_cascade(label=T("menu_edit"), menu=self.edit_menu)

        self.config(menu=self.menubar)

    def _rebuild_menu(self):
        self.config(menu=None)
        self._build_menu()

    # ---------- UI ----------
    def _build_ui(self):
        container = ttk.Frame(self, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        chat_frame = ttk.Frame(container)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_text = tk.Text(
            chat_frame, wrap=tk.WORD, state=tk.DISABLED,
            font=("Segoe UI", 10), padx=6, pady=6
        )
        self.chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(chat_frame, command=self.chat_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_text.configure(yscrollcommand=scroll.set)

        self.chat_text.tag_configure("timestamp", foreground="#888888", spacing1=2, spacing3=6)
        self.chat_text.tag_configure("user", foreground="#0b3d91")
        self.chat_text.tag_configure("assistant", foreground="#2d7d2e")
        italic_font = tkFont.Font(font=("Segoe UI", 10, "italic"))
        self.chat_text.tag_configure("info", foreground="#666666", font=italic_font)

        input_frame = ttk.Frame(container)
        input_frame.pack(fill=tk.X, pady=(8, 0))

        self.entry = tk.Text(input_frame, height=3, font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        buttons = ttk.Frame(input_frame)
        buttons.pack(side=tk.RIGHT)

        self.send_btn = ttk.Button(buttons, text=T("btn_send"), command=self.on_send_clicked)
        self.send_btn.pack(side=tk.TOP, fill=tk.X)

        self.stop_btn = ttk.Button(buttons, text=T("btn_stop"), command=self.on_stop_clicked)
        self.stop_btn.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        self.status_var = tk.StringVar(value=T("status_ready"))
        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(8, 2))
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _bind_shortcuts(self):
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", lambda e: None)

    def _on_return(self, event):
        if event.state & 0x0001:
            return
        self.on_send_clicked()
        return "break"

    # ---------- LLM ----------
    def _init_llm(self):
        model_path = self.cfg.get("model_path")
        if not model_path or not os.path.exists(model_path):
            self._append_info(T("status_no_model"))
            self._set_status(T("status_no_model"))
            self.llm = None
            return
        try:
            self._set_status(T("restarting_llm"))
            self.llm = LocalChatLLM(
                model_path=model_path,
                ctx_size=self.cfg.get("ctx_size", 4096),
                gpu_layers=self.cfg.get("gpu_layers", 0),
                temperature=self.cfg.get("temperature", 0.7),
                top_p=self.cfg.get("top_p", 0.95),
                repeat_penalty=self.cfg.get("repeat_penalty", 1.1),
                n_batch=self.cfg.get("n_batch", 256),
            )
            self._set_status(T("status_ready"))
        except Exception as e:
            self._append_info(f"⚠ {T('llm_error')} {e}")
            self._set_status(T("llm_error"))
            self.llm = None

    # ---------- Chat helpers ----------
    def _now(self) -> str:
        return datetime.now().strftime("%H:%M")

    def _append_text(self, text: str, tag=None):
        self.chat_text.configure(state=tk.NORMAL)
        if tag:
            self.chat_text.insert(tk.END, text, tag)
        else:
            self.chat_text.insert(tk.END, text)
        self.chat_text.see(tk.END)
        self.chat_text.configure(state=tk.DISABLED)

    def _append_line(self, text: str = "", tag=None):
        self._append_text(text + "\n", tag=tag)

    def _append_user(self, message: str):
        self._append_text(f"[{self._now()}] {T('me_label')}\n", tag="timestamp")
        self._append_line(message, tag="user")

    def _append_assistant_header(self):
        self._append_text(f"[{self._now()}] {T('agent_label')}\n", tag="timestamp")

    def _append_assistant_token(self, token: str):
        self.chat_text.configure(state=tk.NORMAL)
        self.chat_text.insert(tk.END, token, "assistant")
        self.chat_text.see(tk.END)
        self.chat_text.configure(state=tk.DISABLED)

    def _append_info(self, message: str):
        self._append_line(message, tag="info")

    def _set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    # ---------- Actions ----------
    def on_send_clicked(self):
        if self.generating:
            return
        if self.llm is None:
            self._append_info(T("status_no_model"))
            return
        user = self.entry.get("1.0", tk.END).strip()
        if not user:
            return
        self.entry.delete("1.0", tk.END)
        self._append_user(user)
        self.history.append({"role": "user", "content": user})
        self._start_generation_thread()

    def on_stop_clicked(self):
        self.generating = False
        self._set_status(T("status_stop_req"))

    def _start_generation_thread(self):
        self.generating = True
        self.current_reply_parts = []
        self._append_assistant_header()
        self._set_status(T("status_thinking"))
        self.send_btn.configure(state=tk.DISABLED)

        t = threading.Thread(target=self._generate_stream, daemon=True)
        t.start()

    def _generate_stream(self):
        try:
            for tok in self.llm.chat_stream(self.history, max_tokens=640):
                if not self.generating:
                    break
                self.current_reply_parts.append(tok)
                self.after(0, self._append_assistant_token, tok)
            reply = "".join(self.current_reply_parts).strip()
            if reply:
                self.history.append({"role": "assistant", "content": reply})
                if len(self.history) > 1 + (MAX_HISTORY_TURNS * 2):
                    self.history = [self.history[0]] + self.history[-(MAX_HISTORY_TURNS * 2):]
        except Exception as e:
            self.after(0, self._append_info, f"⚠ {T('llm_error')} {e}")
        finally:
            self.generating = False
            self.after(0, self._set_status, T("status_ready"))
            self.after(0, lambda: self.send_btn.configure(state=tk.NORMAL))

    # ---------- File menu ----------
    def _save_chat(self):
        content = self.chat_text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo(T("dlg_save_title"), T("dlg_chat_empty"))
            return
        path = filedialog.asksaveasfilename(
            title=T("dlg_save_title"),
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            initialfile=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo(T("dlg_save_title"), f"{T('dlg_saved_to')}\n{path}")
        except Exception as e:
            messagebox.showerror(T("dlg_save_title"), f"{T('dlg_err_saving')} {e}")

    # ---------- Settings ----------
    def _open_settings(self):
        SettingsWindow(self)


class SettingsWindow(tk.Toplevel):
    def __init__(self, parent: ChatGUI):
        super().__init__(parent)
        self.parent = parent
        self.cfg = dict(parent.cfg)
        self.title(T("settings_title"))
        self.geometry("640x460")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_language_tab()
        self._build_models_tab()
        self._build_appearance_tab()

        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(bottom, text=T("close"), command=self._close).pack(side=tk.RIGHT)

    # ---- Language ----
    def _build_language_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=T("tab_language"))

        row = ttk.Frame(tab)
        row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row, text=T("language_choose")).pack(side=tk.LEFT)

        self.lang_var = tk.StringVar(value=self.parent.lang)
        choices = [f"{code} – {label}" for code, label in AVAILABLE_LANGUAGES]
        self.lang_combo = ttk.Combobox(row, state="readonly", values=choices)
        # mostra solo codice nel field
        self.lang_combo.set(self.parent.lang)
        self.lang_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btn_row, text=T("apply_use"), command=self._apply_language).pack(side=tk.LEFT)

    def _apply_language(self):
        raw = self.lang_combo.get()
        new_lang = raw.split(" – ")[0] if " – " in raw else raw
        if new_lang not in dict(AVAILABLE_LANGUAGES):
            return
        self.parent.lang = new_lang
        set_language(new_lang)
        self.parent._refresh_system_prompt_in_cfg()
        self.parent._apply_language_to_gui()
        self._rebuild_models_tab()
        # riapplica tema corrente (in caso stili si resettino)
        self.parent._apply_theme(self.parent.theme)
        messagebox.showinfo(T("settings_title"), T("settings_applied"))

    # ---- Models ----
    def _build_models_tab(self):
        self.models_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.models_tab, text=T("tab_models"))
        self._populate_models_tab()

    def _rebuild_models_tab(self):
        idx = self.notebook.index(self.models_tab)
        self.notebook.forget(idx)
        self._build_models_tab()
        self.notebook.select(self.models_tab)

    def _populate_models_tab(self):
        lang = self.parent.lang
        self.catalog = models_for(lang)

        selector_frame = ttk.Frame(self.models_tab)
        selector_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(selector_frame, text=T("choose_model")).pack(side=tk.LEFT)

        self.model_var = tk.StringVar()
        model_names = [m["name"] for m in self.catalog]
        self.combo = ttk.Combobox(selector_frame, state="readonly", values=model_names, textvariable=self.model_var)
        self.combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        cur_path = self.parent.cfg.get("model_path", "")
        cur_filename = os.path.basename(cur_path) if cur_path else ""
        preselect_index = 0
        for i, m in enumerate(self.catalog):
            if m["filename"] == cur_filename:
                preselect_index = i
                break
        self.combo.current(preselect_index)

        info_frame = ttk.Frame(self.models_tab)
        info_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.status_lbl = ttk.Label(info_frame, text="")
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(info_frame, text=T("check"), command=self._check_selected).pack(side=tk.RIGHT)

        prog_frame = ttk.Frame(self.models_tab)
        prog_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.prog = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate")
        self.prog.pack(fill=tk.X)

        actions = ttk.Frame(self.models_tab)
        actions.pack(fill=tk.X, padx=8, pady=(8, 8))

        self.download_btn = ttk.Button(actions, text=T("download"), command=self._download_selected)
        self.download_btn.pack(side=tk.LEFT)

        self.apply_btn = ttk.Button(actions, text=T("apply_use"), command=self._apply_selected)
        self.apply_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._check_selected()

    def _selected_model(self):
        name = self.model_var.get()
        for m in self.catalog:
            if m["name"] == name:
                return m
        return self.catalog[0]

    def _path_for(self, filename: str) -> str:
        models_dir = self.parent.cfg.get("models_dir")
        if not models_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            self.parent.cfg["models_dir"] = models_dir
        return os.path.join(models_dir, filename)

    def _check_selected(self):
        m = self._selected_model()
        path = self._path_for(m["filename"])
        if os.path.exists(path):
            self.status_lbl.config(text=f"✓ {T('download_done')} {path}")
            self.download_btn.config(state=tk.DISABLED)
            self.apply_btn.config(state=tk.NORMAL)
        else:
            self.status_lbl.config(text=f"✗ {T('not_present')} {path}")
            self.download_btn.config(state=tk.NORMAL)
            self.apply_btn.config(state=tk.DISABLED)
        self.prog.config(value=0, maximum=100, mode="determinate")

    def _download_selected(self):
        m = self._selected_model()
        url = m["url"]
        dest = self._path_for(m["filename"])
        self.download_btn.config(state=tk.DISABLED)
        self.apply_btn.config(state=tk.DISABLED)
        self.prog.config(value=0, mode="determinate")
        self.status_lbl.config(text=T("downloading"))
        t = threading.Thread(target=self._download_thread, args=(url, dest), daemon=True)
        t.start()

    def _download_thread(self, url: str, dest: str):
        tmp = dest + ".part"
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = r.headers.get("content-length")
                total = int(total) if total is not None else None
                if total is None:
                    self.after(0, self._set_progress_indeterminate)
                downloaded = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = int(downloaded * 100 / total)
                                self.after(0, self._set_progress, pct)
            os.replace(tmp, dest)
            self.after(0, self._download_ok, dest)
        except Exception as e:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            self.after(0, self._download_fail, str(e))

    def _set_progress(self, pct: int):
        self.prog.config(mode="determinate", value=pct)

    def _set_progress_indeterminate(self):
        self.prog.config(mode="indeterminate")
        self.prog.start(10)

    def _download_ok(self, dest: str):
        self.prog.stop()
        self.prog.config(mode="determinate", value=100)
        self.status_lbl.config(text=f"✓ {T('download_done')} {dest}")
        self.download_btn.config(state=tk.DISABLED)
        self.apply_btn.config(state=tk.NORMAL)

    def _download_fail(self, err: str):
        self.prog.stop()
        self.prog.config(mode="determinate", value=0)
        self.status_lbl.config(text=f"✗ {T('download_fail')} {err}")
        self.download_btn.config(state=tk.NORMAL)
        self.apply_btn.config(state=tk.DISABLED)

    def _apply_selected(self):
        m = self._selected_model()
        new_path = self._path_for(m["filename"])
        if not os.path.exists(new_path):
            messagebox.showwarning(T("settings_title"), T("download_first"))
            return

        self.parent.cfg["model_url"] = m["url"]
        self.parent.cfg["model_path"] = new_path
        self.parent._refresh_system_prompt_in_cfg()

        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.parent.cfg, f, indent=2)
        except Exception as e:
            messagebox.showerror(T("settings_title"), f"{T('dlg_err_saving')} {e}")
            return

        self.parent._set_status(T("restarting_llm"))
        self.parent.llm = None
        try:
            self.parent._init_llm()
            if self.parent.llm:
                sys_msg = {"role": "system", "content": self.parent.cfg.get("system_prompt", get_system_prompt())}
                self.parent.history = [sys_msg]
                self.parent._append_info(f"{T('model_applied')} {m['name']}")
                messagebox.showinfo(T("settings_title"), T("settings_applied"))
        except Exception as e:
            messagebox.showerror(T("settings_title"), f"{T('llm_error')} {e}")

    # ---- Appearance / Theme ----
    def _build_appearance_tab(self):
        self.appearance_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.appearance_tab, text=T("tab_appearance"))

        row = ttk.Frame(self.appearance_tab)
        row.pack(fill=tk.X, padx=8, pady=8)
        ttk.Label(row, text=T("theme_choose")).pack(side=tk.LEFT)

        self.theme_var = tk.StringVar(value=self.parent.theme)
        radios = ttk.Frame(self.appearance_tab)
        radios.pack(fill=tk.X, padx=8, pady=8)

        ttk.Radiobutton(radios, text=T("theme_dark"), value="dark", variable=self.theme_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(radios, text=T("theme_light"), value="light", variable=self.theme_var).pack(side=tk.LEFT)

        btns = ttk.Frame(self.appearance_tab)
        btns.pack(fill=tk.X, padx=8, pady=(8, 8))
        ttk.Button(btns, text=T("apply"), command=self._apply_theme).pack(side=tk.LEFT)

    def _apply_theme(self):
        new_theme = self.theme_var.get()
        if new_theme not in ("dark", "light"):
            return
        self.parent._apply_theme(new_theme)

    def _close(self):
        self.destroy()


if __name__ == "__main__":
    app = ChatGUI()
    app.mainloop()

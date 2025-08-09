# gui_Style.py
# Gestione tema Dark/Light per la GUI Tk/ttk

from tkinter import ttk

def apply_theme(app, theme: str, chat_text, entry, config_path: str = None, cfg: dict = None):
    """
    Applica tema 'dark' o 'light' ai principali widget.
    app: Tk root
    chat_text, entry: widget Text da tematizzare
    Se cfg e config_path sono forniti, persiste il tema in config.json
    """
    if cfg is not None:
        cfg["theme"] = theme
        if config_path:
            try:
                import json
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass

    style = ttk.Style(app)

    if theme == "dark":
        bg = "#1e1e1e"
        fg = "#e6e6e6"
        subbg = "#2a2a2a"
        accent = "#3a3a3a"
        highlight = "#0b84ff"

        app.configure(bg=bg)
        style.theme_use("clam")
        style.configure(".", background=bg, foreground=fg)

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background=subbg, foreground=fg, padding=6)
        style.map("TButton", background=[("active", accent)])

        style.configure("TEntry", fieldbackground=subbg, foreground=fg)
        style.configure("TCombobox", fieldbackground=subbg, foreground=fg, background=subbg)
        style.map("TCombobox", fieldbackground=[("readonly", subbg)])

        style.configure("TNotebook", background=bg, tabposition="n")
        style.configure("TNotebook.Tab", background=subbg, foreground=fg, padding=(8, 4))
        style.map("TNotebook.Tab",
                  background=[("selected", accent)],
                  foreground=[("selected", fg)])

        style.configure("TProgressbar", background=highlight, troughcolor=subbg)

        chat_text.configure(bg=bg, fg=fg, insertbackground=fg)
        entry.configure(bg=subbg, fg=fg, insertbackground=fg)

    else:  # light
        bg = "#ffffff"
        fg = "#1a1a1a"
        subbg = "#f2f2f2"
        accent = "#e6e6e6"
        highlight = "#0b84ff"

        app.configure(bg=bg)
        style.theme_use("clam")
        style.configure(".", background=bg, foreground=fg)

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background=subbg, foreground=fg, padding=6)
        style.map("TButton", background=[("active", accent)])

        style.configure("TEntry", fieldbackground="#ffffff", foreground=fg)
        style.configure("TCombobox", fieldbackground="#ffffff", foreground=fg, background="#ffffff")
        style.map("TCombobox", fieldbackground=[("readonly", "#ffffff")])

        style.configure("TNotebook", background=bg, tabposition="n")
        style.configure("TNotebook.Tab", background=subbg, foreground=fg, padding=(8, 4))
        style.map("TNotebook.Tab",
                  background=[("selected", "#dddddd")],
                  foreground=[("selected", fg)])

        style.configure("TProgressbar", background=highlight, troughcolor=subbg)

        chat_text.configure(bg="#ffffff", fg=fg, insertbackground=fg)
        entry.configure(bg="#ffffff", fg=fg, insertbackground=fg)

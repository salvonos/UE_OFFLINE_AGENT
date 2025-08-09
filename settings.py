import tkinter as tk
from tkinter import ttk

class SettingsPanel(tk.Toplevel):
    def __init__(self, parent, image_generator):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("300x200")
        self.image_generator = image_generator

        tk.Label(self, text="Image Presets:").pack(pady=5)
        self.preset_var = tk.StringVar(value="Alta qualità")
        self.dropdown = ttk.Combobox(
            self, textvariable=self.preset_var, values=["Alta qualità", "Drammatico", "Bozza veloce"]
        )
        self.dropdown.pack(pady=5)

        tk.Button(self, text="Close", command=self.destroy).pack(pady=10)

    def get_preset(self):
        return self.preset_var.get()

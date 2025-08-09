import os
import subprocess
import sys

def open_folder(path: str):
    """Apre la cartella contenente il file/immagine."""
    folder = os.path.dirname(path) if os.path.isfile(path) else path
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", folder])
    elif os.name == "nt":
        os.startfile(folder)
    elif os.name == "posix":
        subprocess.run(["xdg-open", folder])

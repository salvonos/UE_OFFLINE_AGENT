# voice_rec.py
# Registrazione microfono + riconoscimento vocale locale (Vosk)

import os
import queue
import threading
import zipfile
import shutil
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

@dataclass
class VoskConfig:
    model_dir: str        # cartella dove sta il modello Vosk estratto
    sample_rate: int = 16000
    blocksize: int = 8000  # ~0.5s a 16kHz mono

class VoskRecognizer:
    def __init__(self, cfg: VoskConfig):
        if not os.path.isdir(cfg.model_dir):
            raise FileNotFoundError(f"Vosk model folder not found: {cfg.model_dir}")
        self.cfg = cfg
        self.model = Model(cfg.model_dir)
        self.rec = KaldiRecognizer(self.model, cfg.sample_rate)
        self.rec.SetWords(True)

    def accept(self, pcm_bytes: bytes) -> Optional[str]:
        """Ritorna partial/final text se disponibile, altrimenti None."""
        if self.rec.AcceptWaveform(pcm_bytes):
            j = self.rec.Result()  # final
        else:
            j = self.rec.PartialResult()  # partial
        return j

class MicRecorder:
    """
    Gestisce lo stream microfono e invia chunk PCM 16k mono a un callback.
    """
    def __init__(self, sample_rate: int = 16000, blocksize: int = 8000):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.q = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.running = False

    def _callback(self, indata, frames, time, status):
        if status:
            # status warnings (xruns, etc.)
            pass
        # Converti in int16 mono
        data = (indata * 32767).astype(np.int16)
        if data.ndim > 1:
            data = data[:, 0]
        self.q.put(data.tobytes())

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def read(self, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.stream = None

def download_and_extract_zip(url: str, dest_dir: str, progress_cb: Optional[Callable[[int], None]] = None):
    """
    Scarica uno zip in dest_dir e lo estrae. Se progress_cb è dato, riceve percentuali 0..100.
    """
    import requests, tempfile
    os.makedirs(dest_dir, exist_ok=True)
    tmp_zip = os.path.join(dest_dir, "vosk_tmp.zip")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and progress_cb:
                        progress_cb(int(downloaded * 100 / total))
    # Estrai
    with zipfile.ZipFile(tmp_zip, "r") as z:
        z.extractall(dest_dir)
    os.remove(tmp_zip)

def ensure_vosk_model_folder(base_models_dir: str, lang_code: str) -> str:
    """
    Restituisce il percorso della cartella modello Vosk per la lingua (se esiste), altrimenti path dove verrà estratto.
    Non scarica automaticamente.
    """
    # convenzione: cartella "vosk-{lang}" sotto models/
    return os.path.join(base_models_dir, f"vosk-{lang_code}")

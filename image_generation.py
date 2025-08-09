import threading
from PIL import ImageTk
from helpers_gui import open_folder
from main import generate_image, ensure_sd_pipeline

class ImageGenerator:
    """
    Gestisce la generazione con anteprima live dentro la chat.
    """
    def __init__(self, cfg, chat_widget):
        self.cfg = cfg
        self.chat_widget = chat_widget
        self.last_seed = None
        self.last_prompt = None

    def generate(self, prompt, preset=None):
        """Genera immagine e la mostra in chat con anteprima live."""
        ensure_sd_pipeline(self.cfg)

        # preset opzionali
        steps = 4
        guidance = 0.0
        if preset:
            if preset == "Alta qualità":
                steps, guidance = 12, 7.5
            elif preset == "Drammatico":
                steps, guidance = 8, 6.0
            elif preset == "Bozza veloce":
                steps, guidance = 2, 3.0

        card = self.chat_widget.create_image_progress_card(title="Generazione immagine…")

        def _progress_cb(pct, pil_img):
            def _ui():
                if pil_img is not None:
                    tkimg = ImageTk.PhotoImage(pil_img)
                    card.update_image(tkimg)  # evita GC tramite ref interna
                card.update_progress(pct)
            self.chat_widget.after(0, _ui)

        def _worker():
            try:
                path = generate_image(
                    self.cfg,
                    prompt,
                    steps=steps,
                    guidance=guidance,
                    progress_cb=_progress_cb
                )
                self.last_prompt = prompt
                # opzionale: se salvi il seed in cfg altrove, puoi aggiornarlo qui
                self.chat_widget.after(0, lambda: card.finish(path))
            except Exception as e:
                self.chat_widget.after(0, lambda: card.fail(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def variations(self):
        """Esempio: rigenera con stesso prompt/seed (se disponibile)."""
        if not self.last_prompt:
            self.chat_widget.insert_text("[Nessuna immagine precedente]")
            return
        ensure_sd_pipeline(self.cfg)

        card = self.chat_widget.create_image_progress_card(title="Variazione immagine…")

        def _progress_cb(pct, pil_img):
            def _ui():
                if pil_img is not None:
                    tkimg = ImageTk.PhotoImage(pil_img)
                    card.update_image(tkimg)
                card.update_progress(pct)
            self.chat_widget.after(0, _ui)

        def _worker():
            try:
                path = generate_image(
                    self.cfg,
                    self.last_prompt,
                    progress_cb=_progress_cb
                )
                self.chat_widget.after(0, lambda: card.finish(path))
            except Exception as e:
                self.chat_widget.after(0, lambda: card.fail(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

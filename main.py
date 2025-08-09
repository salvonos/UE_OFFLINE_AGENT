import os, sys, json, subprocess, pathlib, time, re
from typing import List, Dict

APP_NAME = "UE_Offline_Agent"

# ---- LLM (GGUF) ----
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"
)
DEFAULT_MODEL_FILENAME = "llama-3.2-3b-instruct-q4_k_m.gguf"

# ---- Cartelle config ----
CONFIG_DIR = os.path.join(pathlib.Path(__file__).parent, f".{APP_NAME.lower()}")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

# ---- Image Gen (diffusers) ----
SD_TURBO_REPO = "stabilityai/sd-turbo"   # repo HF (usato solo al primo run per cache)
SD_LOCAL_DIR  = "sd-turbo"               # cartella dentro models/ per la cache

# ---- Dipendenze Python ----
REQUIRED_PY_PKGS = [
    "requests>=2.31.0",
    "tqdm>=4.66.0",
    "colorama>=0.4.6",
    "llama-cpp-python>=0.2.90",
    # Immagini
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.39.0",
    "accelerate>=0.27.0",
    "safetensors>=0.4.2",
    "Pillow>=10.0.0",
]

# -----------------------------
# Utils / Log
# -----------------------------
def log(msg: str): print(f"[{APP_NAME}] {msg}")
def warn(msg: str): print(f"[{APP_NAME} ⚠] {msg}")

def ensure_packages():
    """
    Verifica/installa le dipendenze richieste.
    """
    try:
        import llama_cpp, requests  # noqa
        from tqdm import tqdm       # noqa
        import colorama             # noqa
        import torch, diffusers, transformers, accelerate, safetensors, PIL  # noqa
        return
    except Exception:
        pass
    log("Installo dipendenze Python… (potrebbe richiedere qualche minuto)")
    pip = [sys.executable, "-m", "pip", "install", "--upgrade"]
    subprocess.check_call(pip + ["pip"])
    subprocess.check_call(pip + REQUIRED_PY_PKGS)

# -----------------------------
# Config
# -----------------------------
def first_run_setup() -> Dict:
    """
    Crea config iniziale e cartelle base accanto allo script.
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    base_dir = pathlib.Path(__file__).parent.resolve()
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    model_path = os.path.join(models_dir, DEFAULT_MODEL_FILENAME)

    cfg = {
        "base_dir": str(base_dir),
        "models_dir": models_dir,
        "outputs_dir": outputs_dir,
        "model_url": DEFAULT_MODEL_URL,
        "model_path": model_path,
        "ctx_size": 4096,
        "gpu_layers": 0,
        "temperature": 0.7,
        "top_p": 0.95,
        "repeat_penalty": 1.1,
        "n_batch": 256,
        "system_prompt": (
            "Sei un assistente offline per aiutare nello sviluppo con Unreal Engine. "
            "Rispondi in modo conciso, pratico e proponi comandi o codice quando utile."
        ),
        # Immagini
        "image_repo": SD_TURBO_REPO,
        "image_local_dir": SD_LOCAL_DIR,
        "image_steps": 4,
        "image_guidance": 0.0,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg

def _upgrade_config(cfg: Dict) -> Dict:
    """
    Aggiunge chiavi mancanti (per retrocompatibilità con vecchi config).
    """
    base_dir = pathlib.Path(__file__).parent.resolve()
    cfg.setdefault("base_dir", str(base_dir))
    cfg.setdefault("models_dir", os.path.join(base_dir, "models"))
    cfg.setdefault("outputs_dir", os.path.join(base_dir, "outputs"))
    os.makedirs(cfg["models_dir"], exist_ok=True)
    os.makedirs(cfg["outputs_dir"], exist_ok=True)

    cfg.setdefault("model_url", DEFAULT_MODEL_URL)
    cfg.setdefault("model_path", os.path.join(cfg["models_dir"], DEFAULT_MODEL_FILENAME))
    cfg.setdefault("ctx_size", 4096)
    cfg.setdefault("gpu_layers", 0)
    cfg.setdefault("temperature", 0.7)
    cfg.setdefault("top_p", 0.95)
    cfg.setdefault("repeat_penalty", 1.1)
    cfg.setdefault("n_batch", 256)
    cfg.setdefault("system_prompt",
                   "Sei un assistente offline per aiutare nello sviluppo con Unreal Engine. "
                   "Rispondi in modo conciso, pratico e proponi comandi o codice quando utile.")

    cfg.setdefault("image_repo", SD_TURBO_REPO)
    cfg.setdefault("image_local_dir", SD_LOCAL_DIR)
    cfg.setdefault("image_steps", 4)
    cfg.setdefault("image_guidance", 0.0)
    return cfg

def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        return first_run_setup()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = _upgrade_config(cfg)
    # salva eventuali upgrade
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass
    return cfg

# -----------------------------
# Download helper
# -----------------------------
def download(url: str, dest: str):
    """
    Scarica un file grande con barra di progresso.
    """
    import requests
    from tqdm import tqdm
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if os.path.exists(dest):
        log(f"Modello già presente: {dest}")
        return
    log(f"Scarico modello:\n{url}\n→ {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("content-length")
        total = int(total) if total is not None else None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if total:
                        bar.update(len(chunk))
                    else:
                        bar.update(len(chunk))
    os.replace(tmp, dest)
    log("Download completato.")

# -----------------------------
# LLM wrapper (llama.cpp)
# -----------------------------
class LocalChatLLM:
    def __init__(self, model_path, ctx_size, gpu_layers, temperature, top_p, repeat_penalty, n_batch):
        from llama_cpp import Llama
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        log("Inizializzo LLM…")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_threads=max(2, os.cpu_count() // 2),
            n_gpu_layers=gpu_layers,
            n_batch=n_batch,
            verbose=False,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty

    def chat_stream(self, messages: List[Dict], max_tokens: int = 512):
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

# -----------------------------
# Image pipeline (diffusers)
# -----------------------------
_SD_PIPELINE = None

def ensure_sd_pipeline(cfg: Dict):
    """
    Scarica (al primo uso) e carica la pipeline SD-Turbo dentro models/.
    Dopo il primo run, funziona anche offline.
    """
    global _SD_PIPELINE
    if _SD_PIPELINE is not None:
        return _SD_PIPELINE

    from diffusers import StableDiffusionPipeline
    import torch

    repo_id = cfg.get("image_repo", SD_TURBO_REPO)
    local_dir = os.path.join(cfg["models_dir"], cfg.get("image_local_dir", SD_LOCAL_DIR))
    os.makedirs(local_dir, exist_ok=True)

    log(f"Inizializzo pipeline immagini ({repo_id})…")
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        use_safetensors=True,
        cache_dir=local_dir,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass

    _SD_PIPELINE = pipe
    log("Pipeline immagini pronta.")
    return _SD_PIPELINE

def generate_image(cfg: Dict, prompt: str, steps: int | None = None, guidance: float | None = None,
                   width: int = 512, height: int = 512, seed: int | None = None) -> str:
    """
    Genera un'immagine dal prompt e salva in outputs/. Restituisce il path del file PNG.
    """
    from PIL import Image
    import torch

    pipe = ensure_sd_pipeline(cfg)
    steps = steps if steps is not None else int(cfg.get("image_steps", 4))
    guidance = guidance if guidance is not None else float(cfg.get("image_guidance", 0.0))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    log(f"Genero immagine: steps={steps}, guidance={guidance}, size={width}x{height}")
    img = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        width=width,
        height=height,
    ).images[0]

    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"img_{ts}.png"
    out_path = os.path.join(cfg["outputs_dir"], fname)
    os.makedirs(cfg["outputs_dir"], exist_ok=True)
    img.save(out_path)
    log(f"Salvata: {out_path}")
    return out_path

# -----------------------------
# Parser comando /image
# -----------------------------
def _parse_image_command(text: str):
    """
    /image [--steps N] [--size WxH] [--seed N|off] [--guidance F] prompt...
    oppure /img ...
    Ritorna dict parametri o None se non è un comando immagine.
    """
    if not text.lower().startswith(("/image", "/img")):
        return None
    parts = text.split()
    parts = parts[1:]  # togli comando

    steps = None
    width = None
    height = None
    seed = None
    guidance = None

    i = 0
    while i < len(parts):
        p = parts[i].lower()
        if p == "--steps" and i + 1 < len(parts):
            try: steps = int(parts[i+1])
            except: pass
            i += 2; continue
        if p == "--seed" and i + 1 < len(parts):
            if parts[i+1].lower() == "off":
                seed = None
            else:
                try: seed = int(parts[i+1])
                except: pass
            i += 2; continue
        if p == "--size" and i + 1 < len(parts):
            m = re.match(r"(\d+)[xX](\d+)", parts[i+1])
            if m:
                width, height = int(m.group(1)), int(m.group(2))
            i += 2; continue
        if p == "--guidance" and i + 1 < len(parts):
            try: guidance = float(parts[i+1])
            except: pass
            i += 2; continue
        break

    prompt = " ".join(parts[i:]).strip()
    if not prompt:
        prompt = "an image"

    return {
        "prompt": prompt,
        "steps": steps,
        "guidance": guidance,
        "width": width or 512,
        "height": height or 512,
        "seed": seed,
    }

# -----------------------------
# Chat loop (con /image)
# -----------------------------
def run_chat(cfg: Dict):
    # Assicura il modello testuale
    if not os.path.exists(cfg["model_path"]):
        download(cfg["model_url"], cfg["model_path"])

    llm = LocalChatLLM(
        cfg["model_path"], cfg["ctx_size"], cfg["gpu_layers"],
        cfg["temperature"], cfg["top_p"], cfg["repeat_penalty"], cfg["n_batch"]
    )

    system_msg = {"role": "system", "content": cfg["system_prompt"]}
    history: List[Dict] = [system_msg]

    log("Chat pronta. Usa /image (o /img) per generare immagini. Digita 'exit' per uscire.\n")

    while True:
        try:
            user = input("Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(); break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        # --- comando immagine ---
        img_cmd = _parse_image_command(user)
        if img_cmd:
            ensure_sd_pipeline(cfg)  # inizializza pipeline (scarica al primo uso)
            path = generate_image(
                cfg,
                prompt=img_cmd["prompt"],
                steps=img_cmd["steps"],
                guidance=img_cmd["guidance"],
                width=img_cmd["width"],
                height=img_cmd["height"],
                seed=img_cmd["seed"],
            )
            print(f"🖼  Salvata in: {path}")
            # non mandiamo questo testo al LLM
            continue

        # --- normale chat testuale ---
        history.append({"role": "user", "content": user})

        print("\nAgente: ", end="", flush=True)
        reply_parts = []
        try:
            for tok in llm.chat_stream(history, max_tokens=640):
                print(tok, end="", flush=True)
                reply_parts.append(tok)
        except Exception as e:
            warn(f"Errore generazione: {e}")
            print()
            continue

        print("\n")
        reply = "".join(reply_parts)
        history.append({"role": "assistant", "content": reply})

        # tronca contesto (system + ultimi 24 turni)
        if len(history) > 25:
            history = [history[0]] + history[-24:]

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_packages()
    cfg = load_config()
    run_chat(cfg)

if __name__ == "__main__":
    main()

import os, sys, json, subprocess, pathlib
from typing import List, Dict

APP_NAME = "UE_Offline_Agent"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true"
)
DEFAULT_MODEL_FILENAME = "llama-3.2-3b-instruct-q4_k_m.gguf"
CONFIG_DIR = os.path.join(pathlib.Path(__file__).parent, f".{APP_NAME.lower()}")  # nella root script
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

REQUIRED_PY_PKGS = [
    "requests>=2.31.0",
    "tqdm>=4.66.0",
    "colorama>=0.4.6",
    "llama-cpp-python>=0.2.90",
    # +++ voce
    "vosk>=0.3.45",
    "sounddevice>=0.4.6",
    "numpy>=1.24.0",
]

def log(msg: str):
    print(f"[{APP_NAME}] {msg}")

def warn(msg: str):
    print(f"[{APP_NAME} ⚠] {msg}")

def ensure_packages():
    try:
        import llama_cpp, requests
        from tqdm import tqdm
        import colorama
        return
    except Exception:
        pass
    log("Installo dipendenze Python…")
    pip = [sys.executable, "-m", "pip", "install", "--upgrade"]
    subprocess.check_call(pip + ["pip"])
    subprocess.check_call(pip + REQUIRED_PY_PKGS)

def first_run_setup() -> Dict:
    os.makedirs(CONFIG_DIR, exist_ok=True)

    base_dir = pathlib.Path(__file__).parent.resolve()  # cartella script
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, DEFAULT_MODEL_FILENAME)

    cfg = {
        "base_dir": str(base_dir),
        "models_dir": models_dir,
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
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg

def load_config() -> Dict:
    if not os.path.exists(CONFIG_PATH):
        return first_run_setup()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def download(url: str, dest: str):
    import requests
    from tqdm import tqdm
    # Assicura che la cartella di destinazione esista
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
                    bar.update(len(chunk))
    os.replace(tmp, dest)
    log("Download completato.")

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

def run_chat(cfg: Dict):
    if not os.path.exists(cfg["model_path"]):
        download(cfg["model_url"], cfg["model_path"])
    llm = LocalChatLLM(
        cfg["model_path"],
        cfg["ctx_size"],
        cfg["gpu_layers"],
        cfg["temperature"],
        cfg["top_p"],
        cfg["repeat_penalty"],
        cfg["n_batch"],
    )
    system_msg = {"role": "system", "content": cfg["system_prompt"]}
    history: List[Dict] = [system_msg]
    log("Chat pronta. Digita 'exit' per uscire.\n")
    while True:
        try:
            user = input("Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user})
        print("\nAgente: ", end="", flush=True)
        reply_parts = []
        for tok in llm.chat_stream(history, max_tokens=640):
            print(tok, end="", flush=True)
            reply_parts.append(tok)
        print("\n")
        reply = "".join(reply_parts)
        history.append({"role": "assistant", "content": reply})
        if len(history) > 25:
            history = [history[0]] + history[-24:]

def main():
    ensure_packages()
    cfg = load_config()
    run_chat(cfg)

if __name__ == "__main__":
    main()

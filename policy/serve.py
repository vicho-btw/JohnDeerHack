"""
serve.py — servidor de politica Pi-0 (PRUEBA 3, lado pod).

Envuelve la inferencia de pi0 (port PyTorch de openpi via LeRobot) en FastAPI.
- Carga el modelo UNA sola vez al arrancar (startup).
- POST /infer recibe el JSON del contrato (observation_contract.Observation.to_json):
    base_rgb / wrist_rgb : RGB 224x224 uint8 en base64
    state                : 14 floats
    prompt               : string (ingles)
  Decodifica con Observation.from_json (identico a la pista de percepcion).
- Devuelve {"action_chunk": [[...14...] x 50]}  (trayectoria 50x14 del contrato).

Arranque:
    PYTHONPATH=/workspace python policy/serve.py
    # o: uvicorn policy.serve:app --host 0.0.0.0 --port 8000

Puerto: escucha en 0.0.0.0:8000 (ver nota de vast.ai en el reporte / curl abajo).
"""
import os
import sys
import threading

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# el contrato vive en la raiz del repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observation_contract import Observation, STATE_DIM, HORIZON

from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import OBS_STATE

# ---- config ----
CKPT = os.environ.get("PI0_CKPT", "/workspace/ckpts/pi0_base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
# tokenizer PaliGemma oficial esta gated; mirror no-gated con vocab identico
TOKENIZER_MIRROR = os.environ.get("PI0_TOKENIZER", "Shakalaka/paligemma-3b-pt-224")
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title="Pi-0 policy server")

# estado global: modelo cargado una vez; lock para serializar el acceso a la GPU
_M = {"policy": None, "pre": None, "post": None}
_LOCK = threading.Lock()


def load_model():
    print(f"[serve] cargando pi0 desde {CKPT} en {DEVICE} ({DTYPE})...", flush=True)
    policy = PI0Policy.from_pretrained(CKPT, dtype=DTYPE)
    policy.to(DEVICE).eval()
    pre = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_preprocessor.json",
        overrides={"tokenizer_processor": {"tokenizer_name": TOKENIZER_MIRROR}})
    post = PolicyProcessorPipeline.from_pretrained(
        CKPT, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action)
    _M["policy"], _M["pre"], _M["post"] = policy, pre, post
    print("[serve] modelo listo.", flush=True)


@app.on_event("startup")
def _startup():
    load_model()


def _to_chw01(arr):
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)  # HWC [0,1]
    return t.permute(2, 0, 1).contiguous()                # CHW


def run_inference(obs: Observation) -> np.ndarray:
    """Observation -> trayectoria (50, 14) float32."""
    policy, pre, post = _M["policy"], _M["pre"], _M["post"]
    img_keys = list(policy.config.image_features)  # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    imgs = [_to_chw01(obs.base_rgb), _to_chw01(obs.wrist_rgb), _to_chw01(obs.wrist_rgb)]
    batch = {k: im for k, im in zip(img_keys, imgs)}

    # estado: el contrato manda 14-D; pi0 espera max_state_dim (32). Rellenar con ceros.
    state_dim = int(policy.config.input_features[OBS_STATE].shape[0])
    state = np.zeros(state_dim, dtype=np.float32)
    state[:STATE_DIM] = obs.state.astype(np.float32)[:STATE_DIM]
    batch[OBS_STATE] = torch.from_numpy(state)
    batch["task"] = obs.prompt

    with _LOCK, torch.no_grad():
        batch = pre(batch)
        chunk = policy.predict_action_chunk(batch)   # (1, 50, 32)
        chunk = post(chunk)
    traj = chunk.detach().float().cpu().numpy()[0]    # (50, action_dim)
    return traj[:, :STATE_DIM]                         # (50, 14) del contrato


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _M["policy"] is not None,
            "device": DEVICE, "ckpt": CKPT}


@app.post("/infer")
async def infer(request: Request):
    """Body = JSON del contrato (Observation.to_json). Devuelve action_chunk 50x14."""
    body = (await request.body()).decode("utf-8")
    try:
        obs = Observation.from_json(body)   # decodifica identico a la percepcion
        obs.validate()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"observacion invalida: {e}"})
    traj = run_inference(obs)               # (50, 14)
    assert traj.shape == (HORIZON, STATE_DIM), f"shape inesperado {traj.shape}"
    return {"action_chunk": traj.astype(np.float32).tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

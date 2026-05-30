"""
observation_contract.py — LA FUENTE DE VERDAD compartida.

Tanto la pista de percepcion (tu Mac) como la pista de politica (pod con Pi-0)
programan contra esto. Si ambos respetan este formato, el merge es enchufar y listo.

NO cambies los shapes sin avisar a la otra pista.
"""
from dataclasses import dataclass
import numpy as np
import base64
import json


# ---- Formato de la observacion que entra a Pi-0 ----
# images: 3 RGB de 224x224 uint8 (cenital + 2 muneca; en demo la muneca puede ser copia)
# state:  vector de 14 floats (7 joints x 2 brazos). En demo sin robot: dummy fijo.
# prompt: instruccion en lenguaje natural, en INGLES (Pi-0/DROID esperan ingles)
#
# ---- Formato de salida de la politica ----
# action_chunk: np.ndarray shape (50, 14) float32
# Solo se ejecutan los primeros ~5-10 pasos antes de re-planear.

IMG_HW = 224
STATE_DIM = 14
HORIZON = 50

DEFAULT_PROMPT = "pick up the part and align it with the assembly"


@dataclass
class Observation:
    base_rgb: np.ndarray   # (224, 224, 3) uint8 — camara cenital
    wrist_rgb: np.ndarray  # (224, 224, 3) uint8 — opcional, puede ser copia de base
    state: np.ndarray      # (14,) float32
    prompt: str = DEFAULT_PROMPT

    def validate(self):
        assert self.base_rgb.shape == (IMG_HW, IMG_HW, 3), f"base_rgb shape {self.base_rgb.shape}"
        assert self.base_rgb.dtype == np.uint8, f"base_rgb dtype {self.base_rgb.dtype}"
        assert self.wrist_rgb.shape == (IMG_HW, IMG_HW, 3), f"wrist_rgb shape {self.wrist_rgb.shape}"
        assert self.state.shape == (STATE_DIM,), f"state shape {self.state.shape}"
        assert isinstance(self.prompt, str) and len(self.prompt) > 0
        return True

    # ---- Serializacion para mandar al pod por HTTP ----
    def to_json(self) -> str:
        def enc(arr):
            return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")
        return json.dumps({
            "base_rgb": enc(self.base_rgb),
            "wrist_rgb": enc(self.wrist_rgb),
            "state": self.state.astype(np.float32).tolist(),
            "prompt": self.prompt,
        })

    @staticmethod
    def from_json(s: str) -> "Observation":
        d = json.loads(s)
        def dec(b64):
            return np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(IMG_HW, IMG_HW, 3)
        return Observation(
            base_rgb=dec(d["base_rgb"]),
            wrist_rgb=dec(d["wrist_rgb"]),
            state=np.array(d["state"], dtype=np.float32),
            prompt=d["prompt"],
        )


def dummy_observation() -> Observation:
    """Observacion sintetica para probar la politica sin percepcion real."""
    return Observation(
        base_rgb=np.random.randint(0, 255, (IMG_HW, IMG_HW, 3), dtype=np.uint8),
        wrist_rgb=np.random.randint(0, 255, (IMG_HW, IMG_HW, 3), dtype=np.uint8),
        state=np.zeros(STATE_DIM, dtype=np.float32),
        prompt=DEFAULT_PROMPT,
    )


if __name__ == "__main__":
    # PRUEBA 0 — el contrato. Corre en tu Mac con solo numpy instalado.
    obs = dummy_observation()
    obs.validate()
    s = obs.to_json()
    obs2 = Observation.from_json(s)
    obs2.validate()
    assert np.array_equal(obs.base_rgb, obs2.base_rgb)
    print("PRUEBA 0 OK — contrato valido, serializacion round-trip funciona")
    print(f"  base_rgb {obs.base_rgb.shape} {obs.base_rgb.dtype}")
    print(f"  state {obs.state.shape} | prompt: {obs.prompt!r}")

"""
perception/run_video.py — PISTA 1 (tu Mac, sin GPU).

Toma un video POV de ensamble, corre deteccion de objetos (YOLO) + manos (MediaPipe)
+ una heuristica simple de fase, y produce una lista de Observation que cumple el contrato.

PRUEBA 1 (acida): la ultima linea imprime len(observations) y el shape de
observations[0].base_rgb. Si sale (224, 224, 3), percepcion cumple el contrato.

Uso:
    python perception/run_video.py --video data/sample.mp4 --out data/observations.pkl
    python perception/run_video.py --video data/sample.mp4 --max-frames 60 --stride 5
"""
import argparse
import pickle
import sys
import os

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observation_contract import Observation, IMG_HW, STATE_DIM, DEFAULT_PROMPT

# ---- carga perezosa de modelos pesados ----
_yolo = None
_hands = None


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        # yolo11n: el mas chico, corre en CPU. Cambia a yolo11s si tu Mac aguanta.
        _yolo = YOLO("yolo11n.pt")
    return _yolo


def get_hands():
    global _hands
    if _hands is None:
        import mediapipe as mp
        _hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.4,
        )
    return _hands


def detect_objects(frame_bgr):
    """Devuelve lista de (label, conf, (x1,y1,x2,y2))."""
    res = get_yolo()(frame_bgr, verbose=False)[0]
    out = []
    for b in res.boxes:
        cls = int(b.cls[0])
        label = res.names[cls]
        conf = float(b.conf[0])
        xyxy = tuple(map(int, b.xyxy[0].tolist()))
        out.append((label, conf, xyxy))
    return out


def detect_hands(frame_bgr):
    """Devuelve lista de manos; cada mano = np.ndarray (21, 3) en coords normalizadas."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = get_hands().process(rgb)
    hands = []
    if res.multi_hand_landmarks:
        for lm in res.multi_hand_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            hands.append(pts)
    return hands


def infer_phase(objects, hands):
    """
    Heuristica de fase MUY simple (suficiente para el demo).
    En el pitch lo presentas como 'clasificador de fase'; aqui es una regla.
    - sin manos visibles            -> 'idle'
    - manos pero sin objeto cerca   -> 'reaching'
    - manos + objeto                -> 'manipulating'
    Para Assembly101 puedes mapear esto contra las anotaciones reales del dataset.
    """
    if not hands:
        return "idle"
    if hands and not objects:
        return "reaching"
    return "manipulating"


def state_from_hands(hands):
    """
    Deriva un 'state' de 14-D a partir de las manos (proxy, NO es estado de robot real).
    Tomamos la muneca (landmark 0) y algunos dedos de hasta 2 manos para llenar 14 valores.
    En el pitch lo dices honestamente: sin robot, el estado se aproxima de la pose de manos.
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    vals = []
    for h in hands[:2]:
        # muneca + punta de pulgar + punta de indice = 3 puntos x (x,y) ~ 6 vals por mano
        for idx in (0, 4, 8):
            vals.extend([h[idx, 0], h[idx, 1]])
    vals = vals[:STATE_DIM]
    state[:len(vals)] = vals
    return state


def to_obs_image(frame_bgr):
    """Reescala a 224x224 RGB uint8 segun el contrato."""
    img = cv2.resize(frame_bgr, (IMG_HW, IMG_HW))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)


def process_video(path, max_frames=None, stride=1, prompt=DEFAULT_PROMPT):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir el video: {path}")
    observations = []
    annotations = []  # info de overlay por frame, para el dashboard
    fi = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % stride != 0:
            fi += 1
            continue
        objects = detect_objects(frame)
        hands = detect_hands(frame)
        phase = infer_phase(objects, hands)
        state = state_from_hands(hands)
        obs_img = to_obs_image(frame)
        obs = Observation(base_rgb=obs_img, wrist_rgb=obs_img.copy(),
                          state=state, prompt=prompt)
        obs.validate()
        observations.append(obs)
        annotations.append({
            "frame_index": fi,
            "objects": objects,
            "n_hands": len(hands),
            "phase": phase,
        })
        kept += 1
        if max_frames and kept >= max_frames:
            break
        fi += 1
    cap.release()
    return observations, annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="data/observations.pkl")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    obs, ann = process_video(args.video, args.max_frames, args.stride, args.prompt)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"observations": obs, "annotations": ann}, f)

    # PRUEBA 1 (acida)
    print(f"PRUEBA 1 OK — {len(obs)} observaciones generadas")
    print(f"  base_rgb shape: {obs[0].base_rgb.shape}  (esperado (224, 224, 3))")
    print(f"  fases detectadas: {sorted(set(a['phase'] for a in ann))}")
    print(f"  guardado en: {args.out}")


if __name__ == "__main__":
    main()

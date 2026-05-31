"""
perception/run_video.py — PISTA 1 (tu Mac, sin GPU).

Toma un video EGOCENTRICO MONOCROMO (vista POV), corre deteccion de objetos
open-vocabulary (YOLO-World) + manos (MediaPipe) + una heuristica simple de fase,
y produce una lista de Observation que cumple el contrato.

Cambios vs version anterior:
  * Entrada por defecto: data/sample_ego.mp4 (POV monocromo), no sample.mp4.
  * Preprocesado por frame: gris -> 3 canales RGB replicados + CLAHE (mejora de
    contraste). MediaPipe rinde peor en gris; esto ayuda a detectar manos.
  * Objetos open-vocabulary con YOLO-World y --objects "a,b,c": SOLO se buscan
    esas clases (texto libre).
  * Se conserva UNICAMENTE el objeto mas cercano a las manos (el manipulado):
    centro de mano (media de landmarks) -> distancia al centro de cada caja ->
    se queda la mas cercana. Asi el overlay muestra solo el objeto manipulado.

EL CONTRATO Observation NO CAMBIA: base_rgb/wrist_rgb (224,224,3) uint8,
state (14,) float32, prompt str. El filtro de objeto vive en 'annotations',
no en la Observation.

PRUEBA 1 (acida): la ultima linea imprime len(observations) y el shape de
observations[0].base_rgb. Si sale (224, 224, 3), percepcion cumple el contrato.

Uso:
    python perception/run_video.py --objects "toy excavator,screwdriver,hand"
    python perception/run_video.py --video data/sample_ego.mp4 --max-frames 60 --stride 5 \
        --objects "excavator arm,excavator bucket,screw"
"""
import argparse
import pickle
import sys
import os

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observation_contract import Observation, IMG_HW, STATE_DIM, DEFAULT_PROMPT

# Modelo open-vocabulary por defecto. Alternativa: "yolov8s-worldv2.pt".
YOLO_WORLD_WEIGHTS = "yolov8s-world.pt"

# Umbral de confianza para YOLO-World (open-vocab tiende a puntuar bajo;
# como ademas nos quedamos solo con el objeto mas cercano a las manos,
# un umbral bajo es seguro).
YOLO_CONF = 0.05

# Modelo de manos (MediaPipe Tasks). mediapipe>=0.10.3x ya NO trae el modulo
# legacy `solutions`; usamos HandLandmarker. El .task se auto-descarga si falta
# (mismo patron que los pesos de YOLO).
HAND_TASK_PATH = "hand_landmarker.task"
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")

# Clases por defecto para la secuencia 9011-a01 (toy excavator), tomadas de los
# noun_cls reales del dataset (9011-a01_fine_actions.csv). Ajusta con --objects.
DEFAULT_OBJECTS = [
    "screwdriver", "screw", "excavator arm", "bucket",
    "chassis", "track", "cabin", "hand",
]

# ---- carga perezosa de modelos pesados ----
_yolo = None
_hands = None
_object_classes = list(DEFAULT_OBJECTS)  # vocabulario activo de YOLO-World

# CLAHE reutilizable (Contrast Limited Adaptive Histogram Equalization).
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def set_object_classes(classes):
    """Fija el vocabulario open-vocabulary. Lista vacia -> usa DEFAULT_OBJECTS."""
    global _object_classes
    _object_classes = list(classes) if classes else list(DEFAULT_OBJECTS)
    if _yolo is not None:
        _yolo.set_classes(_object_classes)
    return _object_classes


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        # YOLO-World: detecta clases arbitrarias por texto (open-vocabulary).
        _yolo = YOLO(YOLO_WORLD_WEIGHTS)
        _yolo.set_classes(_object_classes)
    return _yolo


def get_hands():
    """HandLandmarker (MediaPipe Tasks API). mediapipe>=0.10.3x no trae solutions."""
    global _hands
    if _hands is None:
        import os, urllib.request
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        if not os.path.exists(HAND_TASK_PATH):
            urllib.request.urlretrieve(HAND_TASK_URL, HAND_TASK_PATH)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
            num_hands=2, min_hand_detection_confidence=0.4,
            running_mode=vision.RunningMode.IMAGE,
        )
        _hands = vision.HandLandmarker.create_from_options(opts)
    return _hands


def preprocess_mono(frame_bgr):
    """Gris -> CLAHE -> 3 canales replicados (uint8).

    Las capturas de cv2 llegan como 3 canales aunque el origen sea monocromo;
    colapsamos a 1 canal, ecualizamos contraste con CLAHE y replicamos a 3
    canales. Como los 3 canales quedan identicos, el orden RGB/BGR es indiferente
    tanto para MediaPipe como para YOLO. Esto SOLO ayuda a la deteccion; el
    base_rgb de la Observation se construye aparte (ver to_obs_image).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    eq = _clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)



# Umbral de croma. Si la diferencia media entre canales por pixel supera esto,
# el frame tiene COLOR real y se usa tal cual (sin CLAHE). Si no, es monocromo
# (gris replicado en 3 canales) y se preprocesa con preprocess_mono.
COLOR_CHROMA_THRESHOLD = 8.0


def is_color(frame_bgr):
    """True si el frame tiene croma real (no es gris replicado en 3 canales)."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return False
    b = frame_bgr[:, :, 0].astype(np.int16)
    g = frame_bgr[:, :, 1].astype(np.int16)
    r = frame_bgr[:, :, 2].astype(np.int16)
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    return float((mx - mn).mean()) > COLOR_CHROMA_THRESHOLD


def preprocess_frame(frame_bgr):
    """Preprocesado adaptativo para la deteccion.

    - Frame YA a color (croma real): se usa tal cual, sin CLAHE (Egocentric-10K).
    - Frame monocromo (gris replicado): CLAHE + 3 canales (preprocess_mono).
    """
    if is_color(frame_bgr):
        return frame_bgr
    return preprocess_mono(frame_bgr)

def detect_objects(proc_bgr):
    """Devuelve lista de (label, conf, (x1,y1,x2,y2)) sobre la imagen preprocesada.

    Con YOLO-World, res.names mapea el indice de clase al texto de --objects.
    """
    res = get_yolo()(proc_bgr, verbose=False, conf=YOLO_CONF)[0]
    out = []
    for b in res.boxes:
        cls = int(b.cls[0])
        label = res.names[cls]
        conf = float(b.conf[0])
        xyxy = tuple(map(int, b.xyxy[0].tolist()))
        out.append((label, conf, xyxy))
    return out


def detect_hands(proc_bgr):
    """Lista de manos; cada mano = np.ndarray (21, 3) normalizada (MediaPipe Tasks)."""
    import mediapipe as mp
    rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = get_hands().detect(mp_img)
    hands = []
    if res.hand_landmarks:
        for lm in res.hand_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            hands.append(pts)
    return hands


def hand_centers_px(hands, hw):
    """Centros (x,y) en pixeles de cada mano: media de sus 21 landmarks."""
    h, w = hw
    centers = []
    for pts in hands:
        cx = float(pts[:, 0].mean()) * w
        cy = float(pts[:, 1].mean()) * h
        centers.append((cx, cy))
    return centers


def nearest_object_to_hands(objects, hands, hw):
    """Conserva SOLO el objeto manipulado: el mas cercano a alguna mano.

    Devuelve una lista con 0 o 1 elemento (misma forma que `objects`), para que
    el overlay muestre una unica caja y infer_phase la consuma igual.

    - Sin objetos -> [].
    - Con objetos pero sin manos -> el de mayor confianza (fallback), asi el
      overlay sigue mostrando el objeto en frames sin mano detectada.
    - Con manos -> distancia minima del centro de cada caja a cualquier mano.
    """
    if not objects:
        return []
    centers = hand_centers_px(hands, hw)
    if not centers:
        return [max(objects, key=lambda o: o[1])]

    def box_center(o):
        x1, y1, x2, y2 = o[2]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def min_dist(o):
        bx, by = box_center(o)
        return min((bx - cx) ** 2 + (by - cy) ** 2 for cx, cy in centers)

    return [min(objects, key=min_dist)]


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
    """Reescala a 224x224 RGB uint8 segun el contrato (frame original, sin CLAHE)."""
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
        proc = preprocess_frame(frame)            # color -> tal cual; mono -> CLAHE+3ch
        objects = detect_objects(proc)
        hands = detect_hands(proc)
        manip = nearest_object_to_hands(objects, hands, frame.shape[:2])  # 0 o 1 objeto
        phase = infer_phase(manip, hands)
        state = state_from_hands(hands)
        obs_img = to_obs_image(frame)
        obs = Observation(base_rgb=obs_img, wrist_rgb=obs_img.copy(),
                          state=state, prompt=prompt)
        obs.validate()
        observations.append(obs)
        annotations.append({
            "frame_index": fi,
            "objects": manip,          # solo el objeto manipulado (0 o 1)
            "n_objects_detected": len(objects),
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
    ap.add_argument("--video", default="data/sample_ego.mp4")
    ap.add_argument("--out", default="data/observations.pkl")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--objects", default="",
        help='Clases a detectar (open-vocab), separadas por coma. '
             'Ej: "metal bracket,screw,excavator part". Vacio -> DEFAULT_OBJECTS.',
    )
    args = ap.parse_args()

    classes = set_object_classes([c.strip() for c in args.objects.split(",") if c.strip()])

    obs, ann = process_video(args.video, args.max_frames, args.stride, args.prompt)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"observations": obs, "annotations": ann}, f)

    # PRUEBA 1 (acida)
    print(f"PRUEBA 1 OK — {len(obs)} observaciones generadas")
    print(f"  base_rgb shape: {obs[0].base_rgb.shape}  (esperado (224, 224, 3))")
    print(f"  clases buscadas: {classes}")
    print(f"  fases detectadas: {sorted(set(a['phase'] for a in ann))}")
    print(f"  guardado en: {args.out}")


if __name__ == "__main__":
    main()

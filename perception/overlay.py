"""
perception/overlay.py — render visual de la percepcion.

Dibuja sobre cada frame (ya mejorado con CLAHE, monocromo):
  * landmarks de las manos (MediaPipe),
  * UNA sola caja: el objeto manipulado (el mas cercano a las manos),
  * la etiqueta de fase.
Produce el video del PANEL IZQUIERDO del dashboard.

Uso:
    python perception/overlay.py --objects "toy excavator,screwdriver,hand"
    python perception/overlay.py --video data/sample_ego.mp4 --out data/overlay.mp4 \
        --objects "excavator arm,excavator bucket,screw"
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.run_video import (
    detect_objects, detect_hands, infer_phase,
    preprocess_mono, preprocess_frame, set_object_classes, nearest_object_to_hands,
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hands(frame, hands):
    h, w = frame.shape[:2]
    for pts in hands:
        px = [(int(x * w), int(y * h)) for x, y, _ in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, px[a], px[b], (80, 220, 120), 2)
        for p in px:
            cv2.circle(frame, p, 3, (40, 160, 90), -1)
    return frame


def draw_object(frame, objects):
    """Dibuja UNA caja: el objeto manipulado (lista de 0 o 1 elemento)."""
    for label, conf, (x1, y1, x2, y2) in objects[:1]:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 130, 240), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 130, 240), 2)
    return frame


def draw_phase(frame, phase):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(frame, f"PHASE: {phase}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/sample_ego.mp4")
    ap.add_argument("--out", default="data/overlay.mp4")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument(
        "--objects", default="",
        help='Clases a detectar (open-vocab), separadas por coma. '
             'Ej: "metal bracket,screw,excavator part". Vacio -> DEFAULT_OBJECTS.',
    )
    args = ap.parse_args()

    set_object_classes([c.strip() for c in args.objects.split(",") if c.strip()])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (w, h))

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % args.stride != 0:
            fi += 1
            continue
        proc = preprocess_frame(frame)             # lienzo monocromo + CLAHE
        objects = detect_objects(proc)
        hands = detect_hands(proc)
        manip = nearest_object_to_hands(objects, hands, frame.shape[:2])  # 0 o 1
        phase = infer_phase(manip, hands)
        canvas = proc                              # dibujamos sobre la vista realzada
        canvas = draw_object(canvas, manip)
        canvas = draw_hands(canvas, hands)
        canvas = draw_phase(canvas, phase)
        vw.write(canvas)
        fi += 1
    cap.release()
    vw.release()
    print(f"overlay guardado en {args.out}")


if __name__ == "__main__":
    main()

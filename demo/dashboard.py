"""
demo/dashboard.py — la pantalla que muestras a las 8 am.

Tres paneles lado a lado:
  IZQUIERDA  : video POV humano con overlay (manos + objeto + fase)
  CENTRO     : brazo Franka simulado ejecutando la trayectoria de Pi-0
  DERECHA    : graficas que demuestran que 'aprendio' la tarea:
               - las 7 curvas de joints de la trayectoria
               - la fase detectada a lo largo del tiempo

Genera un MP4 final. Ese MP4 es tu entregable: si el demo en vivo falla, reproduces esto.

Uso:
    python demo/dashboard.py \
        --overlay data/overlay.mp4 \
        --sim data/sim_frames.pkl \
        --obs data/observations.pkl \
        --traj data/trajectories.pkl \
        --out data/DEMO_FINAL.mp4
"""
import argparse
import pickle
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.robot import normalize_traj

PANEL_H = 480
PANEL_W = 480


def load_overlay_frames(path, n):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (PANEL_W, PANEL_H)))
    cap.release()
    if not frames:
        frames = [np.zeros((PANEL_H, PANEL_W, 3), np.uint8)]
    # estira/recorta para que haya n frames
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx]


def render_plots(traj, phases, t, n):
    """Panel derecho: curvas de joints + barra de fase, dibujado a mano con cv2."""
    img = np.full((PANEL_H, PANEL_W, 3), 245, np.uint8)
    traj = normalize_traj(traj)  # (T, 14)
    T = traj.shape[0]
    # --- curvas de los primeros 7 joints ---
    cv2.putText(img, "Joint trajectory (Pi-0)", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
    x0, y0, pw, ph = 20, 40, PANEL_W - 40, 280
    colors = [(60, 130, 240), (80, 200, 120), (230, 120, 60), (180, 80, 200),
              (60, 200, 200), (200, 60, 120), (120, 120, 60)]
    jmin, jmax = float(traj[:, :7].min()), float(traj[:, :7].max())
    rng = (jmax - jmin) or 1.0
    for j in range(7):
        pts = []
        for k in range(T):
            px = int(x0 + pw * k / max(T - 1, 1))
            py = int(y0 + ph * (1 - (traj[k, j] - jmin) / rng))
            pts.append((px, py))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, colors[j], 1)
    # cursor de tiempo
    cx = int(x0 + pw * t / max(n - 1, 1))
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 1)

    # --- barra de fase ---
    cv2.putText(img, "Detected phase", (12, y0 + ph + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
    phase_now = phases[min(t, len(phases) - 1)] if phases else "n/a"
    cv2.rectangle(img, (20, y0 + ph + 55), (PANEL_W - 20, y0 + ph + 95), (220, 220, 220), -1)
    cv2.putText(img, phase_now, (30, y0 + ph + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 90, 180), 2)
    return img


def label_panel(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (25, 25, 25), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default="data/DEMO_FINAL.mp4")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    with open(args.sim, "rb") as f:
        sim_frames = pickle.load(f)["frames"]
    with open(args.obs, "rb") as f:
        annotations = pickle.load(f)["annotations"]
    with open(args.traj, "rb") as f:
        traj = pickle.load(f)["trajectories"][args.index]

    n = len(sim_frames)
    overlay_frames = load_overlay_frames(args.overlay, n)
    phases = [a["phase"] for a in annotations]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                         (PANEL_W * 3, PANEL_H))
    for t in range(n):
        left = label_panel(overlay_frames[t].copy(), "1. Human POV + perception")
        center = label_panel(cv2.resize(sim_frames[t], (PANEL_W, PANEL_H)),
                             "2. Pi-0 controlling robot (sim)")
        right = label_panel(render_plots(traj, phases, t, n),
                            "3. Learned policy output")
        vw.write(np.hstack([left, center, right]))
    vw.release()
    print(f"DEMO FINAL guardado en {args.out}  ({n} frames, 3 paneles)")
    print(">>> ESTE es tu entregable. Guardalo aunque el demo en vivo falle. <<<")


if __name__ == "__main__":
    main()

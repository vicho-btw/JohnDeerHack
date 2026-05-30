"""
client/call_policy.py — el puente entre tu Mac y el pod.

Manda una Observation al servidor de Pi-0 (corriendo en vast.ai) y recibe la
trayectoria (50, 14). Esta es la PRUEBA 3 (el cruce): si una observacion real
viaja al pod y vuelve trayectoria, el end-to-end esta hecho.

Uso (prueba con dummy):
    python client/call_policy.py --url http://<IP_POD>:<PUERTO>/infer

Uso (con observaciones reales de percepcion):
    python client/call_policy.py --url http://<IP>:<PUERTO>/infer --obs data/observations.pkl
"""
import argparse
import pickle
import sys
import os

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observation_contract import Observation, dummy_observation, HORIZON, STATE_DIM


def call(url, obs: Observation, timeout=120):
    obs.validate()
    r = requests.post(url, data=obs.to_json(),
                      headers={"Content-Type": "application/json"}, timeout=timeout)
    r.raise_for_status()
    traj = np.array(r.json()["action_chunk"], dtype=np.float32)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="http://<IP_POD>:<PUERTO>/infer")
    ap.add_argument("--obs", default=None, help="pickle de percepcion; si falta usa dummy")
    ap.add_argument("--out", default="data/trajectories.pkl")
    args = ap.parse_args()

    if args.obs:
        with open(args.obs, "rb") as f:
            data = pickle.load(f)
        observations = data["observations"]
    else:
        observations = [dummy_observation()]

    trajectories = []
    for i, obs in enumerate(observations):
        traj = call(args.url, obs)
        trajectories.append(traj)
        if i == 0:
            # PRUEBA 3 (acida)
            ok = traj.shape == (HORIZON, STATE_DIM) or traj.shape == (STATE_DIM, HORIZON)
            print(f"PRUEBA 3 {'OK' if ok else 'REVISAR'} — trayectoria recibida shape {traj.shape}")
            print(f"  esperado ({HORIZON}, {STATE_DIM}) o ({STATE_DIM}, {HORIZON})")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(observations)} observaciones procesadas")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"trajectories": trajectories}, f)
    print(f"guardadas {len(trajectories)} trayectorias en {args.out}")


if __name__ == "__main__":
    main()

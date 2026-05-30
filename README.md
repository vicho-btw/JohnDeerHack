# JohnDeerHack — Automatización de ensamble por demostración (VLA)

Pipeline end-to-end: **video POV de ensamble → percepción (manos+objeto+fase) →
Pi-0 genera trayectoria → brazo simulado la ejecuta → dashboard de 3 paneles.**

Tesis del pitch: el cuello de botella de automatizar ensamble no son los robots ni los
modelos (existen, open source) sino **datos de demostración en el formato correcto**.
Construimos la fábrica de datos que hace cualquier estación automatizable. Para producción,
John Deere pone cámaras POV a sus operarios y graba su propio dataset propietario — ese es
el activo.

## Dos pistas en paralelo

- **Pista 1 — TU MAC (sin GPU):** percepción, simulador, dashboard. Todo CPU.
- **Pista 2 — POD vast.ai (GPU):** Pi-0 carga checkpoint y genera trayectorias. Lo maneja
  Claude Code por SSH. Prompts en `policy/PROMPTS.md`.

Ambas hablan vía `observation_contract.py` (la fuente de verdad). Respétalo y el merge es trivial.

## Estrategia de GPU: efímera, no persistente

La GPU (RTX 3090, ~$0.55/h) solo se necesita en dos ventanas: el gate de Pi-0 y generar
trayectorias. ~2-3h reales de GPU = ~$1.50 de los $10. **Apaga/destruye el pod entre ventanas.**
Pon una alarma por si te duermes con el pod encendido.

## Pruebas ácidas (corren aisladas, sin dependencias entre sí)

| Prueba | Dónde | Comando | Pasa si... |
|---|---|---|---|
| 0 — contrato | Mac | `python observation_contract.py` | imprime "PRUEBA 0 OK" |
| 1 — percepción | Mac | `python perception/run_video.py --video data/sample.mp4` | shape (224,224,3) |
| 2 — política | Pod | Prompt 1 de PROMPTS.md | shape (50,14) impreso |
| 3 — el cruce | Mac→Pod | `python client/call_policy.py --url http://IP:PUERTO/infer` | trayectoria recibida |

Solo la Prueba 3 junta ambas pistas. Si 1 y 2 están en verde, el merge toma minutos.

## Orden de operaciones (la noche)

### Ahora mismo (en seco, antes de prender la GPU)
```bash
# 1. clona tu repo, instala deps de Mac
python -m venv venv && source venv/bin/activate
pip install -r requirements-mac.txt

# 2. PRUEBA 0 — valida el contrato (no necesita nada pesado)
python observation_contract.py
```

### Prender el pod + lanzar Claude Code (Pista 2)
```bash
# en el pod, por SSH:
tmux new -s policy
npm install -g @anthropic-ai/claude-code   # si no está; o usa npx
claude
# pega PROMPT 0 (baja Assembly101 → data/sample.mp4)
# luego PROMPT 1 (gate de Pi-0). NO sigas hasta que imprima shape (50,14).
```

### Mientras Pi-0 carga, tú en el Mac (Pista 1)
```bash
# baja data/sample.mp4 del pod (scp) o usa cualquier video POV de ensamble
# PRUEBA 1 — percepción
python perception/run_video.py --video data/sample.mp4 --out data/observations.pkl
# genera el overlay (panel izquierdo)
python perception/overlay.py --video data/sample.mp4 --out data/overlay.mp4
```

### Cuando el gate pasó → Claude Code levanta el servidor (PROMPT 2)
```bash
# en el Mac, PRUEBA 3 — el cruce
python client/call_policy.py --url http://<IP_POD>:<PUERTO>/infer            # dummy primero
python client/call_policy.py --url http://<IP_POD>:<PUERTO>/infer \
    --obs data/observations.pkl --out data/trajectories.pkl                   # luego real
```

### Integración final (Mac, CPU)
```bash
# brazo simulado ejecuta la trayectoria (--gui para verlo en vivo)
python sim/robot.py --traj data/trajectories.pkl --out data/sim_frames.pkl
# dashboard de 3 paneles → tu ENTREGABLE
python demo/dashboard.py \
    --overlay data/overlay.mp4 --sim data/sim_frames.pkl \
    --obs data/observations.pkl --traj data/trajectories.pkl \
    --out data/DEMO_FINAL.mp4
```

## El gate que importa: 4:00 am

**Ten `data/DEMO_FINAL.mp4` grabado.** Si el demo en vivo falla a las 8am, reproduces el MP4.
No te duermas sin ese archivo.

## Si Pi-0 NO carga a tiempo
Usa PROMPT 3 (fallback): un mock con idéntica interfaz `/infer`. El resto del pipeline ni se
entera. Tu demo queda blindado.

## Notas
- Pi-0/DROID esperan prompts en **inglés**.
- Pi-0 da 14-D (2 brazos). El Franka usa los primeros 7 (un brazo). Suficiente para el demo.
- El `state` de 14-D sin robot real es un proxy de la pose de manos — dilo con honestidad en el pitch.
- Assembly101 y LocateAnything son licencia **no-comercial**: ok para hackathon; producción = dataset propio.

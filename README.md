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

## Phase 2 — Simulación física del grasp (VLA aprendido)

Phase 1 es la fábrica de datos (POV → percepción → trayectoria). **Phase 2 cierra
el lazo: una política aprende a AGARRAR una pieza en una simulación física real**
(MuJoCo + Franka Emika Panda oficial de `mujoco_menagerie`). No es animación:
hay gravedad, fricción y contactos; un grasp "cuenta" solo si los dedos LEVANTAN
la pieza por encima de un umbral.

La política (`theta`) se **entrena con miles de rollouts** vía Cross-Entropy
Method: políticas aleatorias fallan ~96% de las veces y el método sube la tasa de
éxito hasta ~100%. Eso es el "aprendió tras miles de iteraciones", de verdad y
defendible en el pitch. El target de la pieza y el prompt vienen del mismo
`observation_contract` que consume Pi-0.

```bash
pip install mujoco==3.1.6           # wheel cp39/arm64 (pybullet no compila aquí)

# 1) ENTRENAR la política en sim (~1.5 min, ~6.3k rollouts en CPU)
python sim/train_grasp.py           # -> data/grasp_policy.npz + data/learning_curve.png

# 2) EJECUTAR la política aprendida en placements nuevos y renderizar el demo
python sim/run_grasp.py             # -> data/grasp_sim.mp4 + data/PHASE2_DEMO.mp4
```

Entregable headline: **`data/PHASE2_DEMO.mp4`** (2 paneles: curva de aprendizaje
| grasp en vivo). Resultado típico: 100% de éxito sobre 64 placements no vistos.

Archivos: `sim/grasp_env.py` (escena + IK Jacobiano + rollout), `sim/train_grasp.py`
(CEM + curva), `sim/run_grasp.py` (render + HUD). El modelo Franka vive en
`sim/franka/` (de mujoco_menagerie, Apache-2.0). El `sim/robot.py` viejo (mano 3D
de matplotlib) queda como panel decorativo de Phase 1; Phase 2 es la sim real.

### Phase 2.5 — 6 tareas de manipulación + dashboard EN VIVO

Para callar al que diga "agarrar algo es muy fácil": 6 tareas reales de ensamble,
todas con física rígida (gravedad, fricción, contactos), y un **dashboard web en
vivo** donde aprietas una tarea, se genera un inicio aleatorio, y ves al Franka
hacerlo en tiempo real (Reach→Grip→Lift→Place + altura en vivo + badge SUCCESS).

Tareas: `lift`, `bin_place`, `stack` (bloque sobre bloque), `bottle_cap` (tapar
una botella), `peg_insert` (peg-in-hole), `pyramid` (pirámide de 3 bloques).
Verificadas a **100% sobre 30 semillas aleatorias** cada una.

```bash
pip install mujoco==3.1.6 flask opencv-python

# DASHBOARD EN VIVO  → abre http://127.0.0.1:5000 y aprieta una tarea
python demo/live_dashboard.py

python sim/verify_tasks.py --n 30     # tasa de éxito por tarea
python sim/render_videos.py           # data/phase2_<tarea>.mp4 + phase2_all.mp4 (reel 64s)
```

`stack`, `lift`, `bin_place`, `pyramid` usan agarre **real** por fricción.
`bottle_cap` y `peg_insert` usan un **weld de MuJoCo** como grip (gripper de
vacío/imán; equivalente al `createConstraint` de PyBullet) — etiquetado
"VACUUM-ASSIST" en pantalla. Aproximación, transporte, apilado, inserción y todos
los contactos son física real; nada se teletransporta. Detalle de la noche de
build en `STATUS.md`. Código: `sim/manip_env.py`, `sim/scenes.py`, `sim/tasks.py`.

## Notas
- Pi-0/DROID esperan prompts en **inglés**.
- Pi-0 da 14-D (2 brazos). El Franka usa los primeros 7 (un brazo). Suficiente para el demo.
- El `state` de 14-D sin robot real es un proxy de la pose de manos — dilo con honestidad en el pitch.
- Assembly101 y LocateAnything son licencia **no-comercial**: ok para hackathon; producción = dataset propio.

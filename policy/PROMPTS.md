# PROMPTS para Claude Code (corren EN EL POD de vast.ai)

Pega estos en orden. No saltes el gate del Prompt 1 antes de seguir.

---

## PROMPT 0 — descargar muestra de Assembly101

> Antes de tocar Pi-0: baja una muestra MINIMA del dataset Assembly101 desde Hugging Face
> (NO de Google Drive, esta siendo retirado segun migracion de mayo 2026). Necesito:
> 1-2 videos de vista EGOCENTRICA de UNA secuencia de ensamble, sus anotaciones de accion
> fina, y si es facil, las poses de mano 3D de esa secuencia. El repo de scripts de descarga
> es `assembly-101/assembly101-download-scripts` pero verifica primero la ruta de Hugging
> Face por la migracion. NO bajes las 513 horas completas — solo lo minimo para un demo.
> Cuando termine, dime las rutas de los archivos y dame un script que reproduzca un frame
> del video con su anotacion de accion correspondiente impresa encima. Convierte un clip
> corto a MP4 y dejalo en `data/sample.mp4` para que yo lo baje a mi Mac.

---

## PROMPT 1 — gate de Pi-0 (EL CRITICO)

> Estoy en un hackathon con deadline a las 8am. Esta maquina es un pod vast.ai con GPU
> RTX 3090 (24GB), Ubuntu 22.04, PyTorch+CUDA preinstalado. Tu unico objetivo en esta fase:
> lograr que el modelo Pi-0 (π0) de Physical Intelligence cargue un checkpoint pre-entrenado
> y genere UNA trayectoria de accion a partir de una observacion sintetica. No me ayudes con
> percepcion ni simulacion todavia.
>
> Secuencia estricta, no saltes pasos:
> 1. Corre `nvidia-smi` y dime que GPU y cuanta VRAM hay.
> 2. Intenta primero el port de PyTorch de openpi via LeRobot (mas facil en este entorno).
>    Si en 45 min no carga, cae al repo oficial `Physical-Intelligence/openpi` en JAX.
> 3. Carga el checkpoint **π0-DROID** (el afinado a manipulacion brazo-sobre-mesa, no el base).
> 4. Construye una observacion sintetica: 3 imagenes RGB de 224x224 random, un prompt de texto
>    "pick up the part and align it with the assembly", y un estado de 14 dimensiones en ceros.
> 5. Corre inferencia y haz print() del shape de la trayectoria devuelta (~ (50,14) o (14,50)).
>
> Cuando el paso 5 imprima un shape valido, PARA y avisame. Ese es el gate. No optimices nada
> mas hasta entonces. Si algo falla, muestrame el traceback completo y tu hipotesis antes del fix.

---

## PROMPT 2 — servidor de inferencia (cuando el gate paso)

> El gate paso: Pi-0 genera trayectorias desde una observacion sintetica. Ahora envuelve la
> inferencia en un servidor HTTP para que mi codigo de percepcion (en otra maquina) le mande
> observaciones reales.
>
> Requisitos:
> 1. FastAPI con endpoint POST `/infer` que reciba JSON con: tres imagenes base64 (campos
>    `base_rgb`, `wrist_rgb` — cada una son los bytes de un array uint8 224x224x3), un array
>    `state` de 14 floats, y un `prompt` string. El formato exacto esta en `observation_contract.py`
>    de mi repo (metodo `to_json`/`from_json`) — usalo para decodificar identico.
> 2. Decodifica, arma la observacion que Pi-0 espera, corre inferencia, y devuelve JSON
>    `{"action_chunk": [[...14...], ... 50 filas ...]}`.
> 3. Que escuche en `0.0.0.0` en un puerto, y dime que puerto debo EXPONER en vast.ai
>    (en la config del pod) para alcanzarlo desde mi Mac.
> 4. Incluye un `curl` de ejemplo o script cliente que mande una observacion dummy e imprima
>    la trayectoria recibida.
>
> Carga el modelo en memoria UNA sola vez al arrancar, no por request. Avisame cuando el curl
> de prueba funcione. Guarda el server en `policy/serve.py`.

---

## PROMPT 3 — FALLBACK (solo si Pi-0 NO carga a tiempo)

> Pi-0 no logro cargar a tiempo y necesito un mock creible para no romper el demo. Crea un
> servidor FastAPI con el MISMO endpoint `/infer` y el MISMO formato de entrada/salida que
> `observation_contract.py`, pero en vez de correr Pi-0, que genere una trayectoria (50,14)
> plausible: interpolacion suave entre el `state` recibido y una pose objetivo, con algo de
> ruido realista para que se vea organica en el simulador. Marca CLARAMENTE en el codigo que
> es un mock para poder cambiarlo por Pi-0 real despues. Guardalo en `policy/serve_mock.py`.

---

## Nota sobre exponer el puerto en vast.ai

Al crear el pod, en la config hay un campo de puertos a exponer (o usas el "Open Ports").
Pidele a Claude Code el numero de puerto del paso 3 del Prompt 2 y asegurate de que ese
puerto este abierto en el pod, o el `client/call_policy.py` de tu Mac no podra alcanzarlo.
Alternativa rapida si el puerto no abre: `ssh -L 8000:localhost:8000 ...` (tunel SSH) y
apuntas el cliente a `http://localhost:8000/infer`.

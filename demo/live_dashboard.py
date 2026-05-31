"""
demo/live_dashboard.py — Phase 2 LIVE dashboard.

Open it in a browser, click a task, and watch the Franka do it in real time on a
fresh randomized layout. Each click rolls a new random seed, so it's never the
same run twice. Live HUD shows the stage (Reach -> Grip -> Lift -> Place), the
live object height, and a SUCCESS / constraint-assist badge.

Architecture: ONE persistent worker thread owns the MuJoCo envs + the OpenGL
renderer (macOS-safe — all GL on one thread). The controllers' `emit` callback
renders each frame, draws the HUD, and publishes the latest JPEG. Flask streams
that buffer as MJPEG (looks live) and exposes /run + /status. Only one rollout
runs at a time.

Run:
    python demo/live_dashboard.py            # then open http://127.0.0.1:5000
"""
import os
import sys
import threading
import queue
import time

import numpy as np
import cv2
from flask import Flask, Response, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.manip_env import ManipEnv
from sim.tasks import TASKS, TASK_ORDER, run_task

FR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim", "franka")
TRACK = {"lift": "part", "bin_place": "part", "stack": "top",
         "bottle_cap": "cap", "peg_insert": "peg", "pyramid": "b2"}
STAGE_COLOR = {"Reach": (60, 170, 230), "Grip": (60, 200, 230), "Lift": (90, 200, 90),
               "Move": (200, 150, 60), "Place": (210, 160, 70), "Settle": (160, 160, 160),
               "Done": (90, 210, 110), "SUCCESS": (90, 210, 110), "idle": (120, 120, 120)}
W, H = 720, 560


def _bar(img, y0, y1, a=0.5):
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (img.shape[1], y1), (16, 16, 20), -1)
    cv2.addWeighted(ov, a, img, 1 - a, 0, img)


class SimWorker(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__()
        self.cmd = queue.Queue()
        self.envs = {}
        self._lock = threading.Lock()
        self.jpeg = self._placeholder()
        self.status = {"task": None, "label": "Ready", "stage": "idle", "mode": "",
                       "z_cm": 0.0, "running": False, "success": None, "seed": None,
                       "criterion": ""}
        self._seed = 7

    # ---- public API (called from Flask threads) ----
    def request(self, task_name):
        if self.status["running"] or task_name not in TASKS:
            return False
        self._seed = (self._seed * 1103515245 + 12345) & 0x7FFFFFFF  # LCG, no RNG dep
        self.cmd.put((task_name, self._seed % 100000))
        return True

    def latest_jpeg(self):
        with self._lock:
            return self.jpeg

    # ---- worker thread ----
    def run(self):
        # pre-load the first env so the idle frame shows the real robot
        try:
            self._render_idle(TASK_ORDER[0])
        except Exception as e:
            print(f"[worker] idle preload failed: {e}", flush=True)
        while True:
            task_name, seed = self.cmd.get()
            try:
                self._run_task(task_name, seed)
            except Exception as e:  # never kill the worker; surface the error
                import traceback; traceback.print_exc()
                self.status.update(running=False, label=f"error: {e}", stage="idle")

    def _get_env(self, name):
        if name not in self.envs:
            env = ManipEnv(os.path.join(FR, f"task_{TASKS[name].scene}.xml"),
                           render_size=(H, W))
            env.frame_every = 6
            self.envs[name] = env
        return self.envs[name]

    def _run_task(self, name, seed):
        task = TASKS[name]
        env = self._get_env(name)
        mode = "VACUUM-ASSIST" if task.default_weld else "REAL GRASP"
        self.status.update(task=name, label=task.label, mode=mode, running=True,
                           success=None, seed=seed, criterion=task.criterion, stage="Reach")
        track = TRACK[name]

        def emit(e):
            z = float(e.body_pos(track)[2])
            self.status["z_cm"] = z * 100
            self.status["stage"] = e.stage
            self._publish(e, name, mode)
            time.sleep(1 / 60.0)  # pace to a watchable real-time-ish speed

        ok, z, text, _ = run_task(env, task, seed=seed, emit=emit)
        self.status.update(running=False, success=bool(ok),
                           stage="SUCCESS" if ok else "done", z_cm=z * 100)
        # hold the final frame with the result badge for a moment
        for _ in range(45):
            self._publish(env, name, mode, final=True, ok=bool(ok))
            time.sleep(1 / 30.0)

    # ---- rendering / HUD ----
    def _publish(self, env, name, mode, final=False, ok=False):
        frame = env.render_frame()
        self._hud(frame, name, mode, final, ok)
        okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if okj:
            with self._lock:
                self.jpeg = buf.tobytes()

    def _hud(self, f, name, mode, final, ok):
        task = TASKS[name]
        st = self.status["stage"]
        _bar(f, 0, 64)
        cv2.putText(f, task.label, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (245, 245, 248), 2, cv2.LINE_AA)
        cv2.putText(f, task.desc, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (170, 200, 230), 1, cv2.LINE_AA)
        cv2.putText(f, mode, (W - 175, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 220, 150) if "REAL" in mode else (220, 200, 120), 1, cv2.LINE_AA)
        _bar(f, H - 56, H)
        cv2.putText(f, f"height: {self.status['z_cm']:5.1f} cm", (14, H - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(f, f"seed {self.status['seed']}   goal: {task.criterion}", (14, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 190), 1, cv2.LINE_AA)
        chip = "SUCCESS" if (final and ok) else ("FAILED" if final else st)
        col = (90, 200, 90) if (final and ok) else ((60, 60, 210) if final else
                                                    STAGE_COLOR.get(st, (150, 150, 150)))
        (tw, _), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.rectangle(f, (W - tw - 28, H - 46), (W - 12, H - 18), col, -1)
        cv2.putText(f, chip, (W - tw - 20, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (20, 20, 20), 2, cv2.LINE_AA)

    def _render_idle(self, name):
        env = self._get_env(name)
        env.home()
        self._publish(env, name, "REAL GRASP" if not TASKS[name].default_weld else "VACUUM-ASSIST")

    def _placeholder(self):
        img = np.full((H, W, 3), 22, np.uint8)
        cv2.putText(img, "Phase 2 - loading Franka...", (W // 2 - 180, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        return cv2.imencode(".jpg", img)[1].tobytes()


WORKER = SimWorker()
app = Flask(__name__)


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Phase 2 - Live Franka</title>
<style>
 body{margin:0;background:#0d0f12;color:#e8eaed;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1080px;margin:0 auto;padding:22px}
 h1{font-size:21px;margin:0 0 2px} .sub{color:#8b9bb0;font-size:13px;margin-bottom:16px}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 .stage{flex:1 1 720px} img{width:100%;border-radius:12px;background:#000;display:block}
 .side{flex:1 1 260px;min-width:240px}
 .btns{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
 button{background:#1a1d23;color:#e8eaed;border:1px solid #2b3038;border-radius:10px;
   padding:13px 10px;font-size:14px;cursor:pointer;text-align:left;transition:.15s}
 button:hover{background:#22262e;border-color:#3a7d44}
 button:disabled{opacity:.4;cursor:default}
 button b{display:block;font-size:14px} button span{font-size:11px;color:#8b9bb0}
 .panel{background:#14171c;border:1px solid #242a32;border-radius:10px;padding:14px;font-size:13px}
 .k{color:#8b9bb0} .v{float:right;font-weight:600}
 .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:700}
 .accent{color:#5fd07a}
</style></head><body><div class=wrap>
 <h1>Phase 2 — Live Franka manipulation <span class=accent>(MuJoCo)</span></h1>
 <div class=sub>Click a task. Each run rolls a new random starting layout and runs real rigid-body physics.</div>
 <div class=row>
  <div class=stage><img id=cam src="/stream"></div>
  <div class=side>
   <div class=btns id=btns></div>
   <div class=panel>
     <div><span class=k>Status</span><span class=v id=st>ready</span></div>
     <div><span class=k>Stage</span><span class=v id=stage>-</span></div>
     <div><span class=k>Height</span><span class=v id=z>-</span></div>
     <div><span class=k>Grip</span><span class=v id=mode>-</span></div>
     <div style="margin-top:10px" id=result></div>
   </div>
  </div>
 </div>
</div>
<script>
const TASKS=__TASKS__;
const btns=document.getElementById('btns');
TASKS.forEach(t=>{const b=document.createElement('button');b.id='b_'+t.name;
  b.innerHTML='<b>'+t.label+'</b><span>'+t.criterion+'</span>';
  b.onclick=()=>fetch('/run/'+t.name,{method:'POST'});btns.appendChild(b);});
async function poll(){try{const s=await (await fetch('/status')).json();
  document.getElementById('st').textContent=s.running?('running '+s.label):(s.label||'ready');
  document.getElementById('stage').textContent=s.stage;
  document.getElementById('z').textContent=s.z_cm.toFixed(1)+' cm';
  document.getElementById('mode').textContent=s.mode||'-';
  const r=document.getElementById('result');
  if(s.success===true)r.innerHTML='<span class=badge style="background:#1f8a4c">SUCCESS</span>';
  else if(s.success===false)r.innerHTML='<span class=badge style="background:#9a3030">FAILED</span>';
  else r.innerHTML='';
  document.querySelectorAll('button').forEach(b=>b.disabled=s.running);
}catch(e){}setTimeout(poll,250);}poll();
</script></body></html>"""


@app.route("/")
def index():
    import json
    tasks = [{"name": n, "label": TASKS[n].label, "criterion": TASKS[n].criterion}
             for n in TASK_ORDER]
    return PAGE.replace("__TASKS__", json.dumps(tasks))


@app.route("/run/<task>", methods=["POST"])
def run(task):
    return jsonify({"ok": WORKER.request(task)})


@app.route("/status")
def status():
    return jsonify(WORKER.status)


@app.route("/stream")
def stream():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            frame = WORKER.latest_jpeg()
            yield boundary + frame + b"\r\n"
            time.sleep(1 / 30.0)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    WORKER.start()
    print("Phase 2 live dashboard -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)

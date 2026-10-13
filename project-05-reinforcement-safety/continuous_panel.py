"""Native teaching panel: MuJoCo live demo, GIFs, training and result inspection."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/"outputs/continuous-20261013-r2"


class ContinuousPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("连续动作PPO · MuJoCo物理实验")
        self.geometry("1060x800")
        self.minsize(1000, 760)
        self.process = self.stream = None
        self.frames, self.frame, self.playing = [], 0, False
        self.folder = BASELINE
        self.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(self)
        style.configure("TButton", padding=(10, 8), font=("Microsoft YaHei UI", 11))
        style.configure("TLabel", font=("Microsoft YaHei UI", 11))
        ttk.Label(self, text="PPO：目标条件化策略 → 速度参考 → 伺服力矩 → 物理运动", font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w", padx=18, pady=14)
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=18)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 18))
        self.picture = ttk.Label(left, anchor="center")
        self.picture.pack(fill="both", expand=True)
        self.metrics = tk.Text(left, height=12, font=("Microsoft YaHei UI", 11), wrap="word", bg="#f8fafc", fg="#0f172a")
        self.metrics.pack(fill="x", pady=10)
        right = ttk.Frame(body, width=320)
        right.pack(side="right", fill="y")
        ttk.Label(right, text="控制器（同一案例条件）").pack(anchor="w")
        self.policy = tk.StringVar(value="ppo_seed_0")
        ttk.Combobox(right, textvariable=self.policy, values=["ppo_seed_0", "ppo_seed_1", "ppo_seed_2", "ppo_seed_3", "ppo_seed_4", "dls", "zero", "random", "untrained"], state="readonly").pack(fill="x", pady=6)
        ttk.Label(right, text="域 / 案例编号（0—23）").pack(anchor="w")
        fields = ttk.Frame(right)
        fields.pack(fill="x", pady=6)
        self.split, self.case = tk.StringVar(value="test"), tk.StringVar(value="0")
        ttk.Combobox(fields, textvariable=self.split, values=["test", "ood"], state="readonly", width=10).pack(side="left")
        ttk.Combobox(fields, textvariable=self.case, values=list(map(str, range(24))), state="readonly", width=8).pack(side="left", padx=8)
        for text, callback in (("运行所选案例 · 实时3D", self.live), ("载入种子0 / DLS / 未训练回放", self.load_replay),
                               ("回放：播放 / 暂停", self.toggle), ("打开PPO知识与操作课", lambda: os.startfile(ROOT/"03-连续动作PPO实操.md")),
                               ("打开GPT与π0.5应用课", lambda: os.startfile(ROOT/"04-GPT6与pi05机器人应用.md")),
                               ("打开原始结果目录", lambda: os.startfile(self.folder))):
            ttk.Button(right, text=text, command=callback).pack(fill="x", pady=(2, 6))
        ttk.Label(right, text="新实验交互步数 / 训练种子").pack(anchor="w", pady=(8, 0))
        entries = ttk.Frame(right)
        entries.pack(fill="x", pady=6)
        self.steps, self.seed = tk.StringVar(value="65536"), tk.StringVar(value="5")
        ttk.Entry(entries, textvariable=self.steps, width=12).pack(side="left")
        ttk.Entry(entries, textvariable=self.seed, width=8).pack(side="left", padx=8)
        self.train_button = ttk.Button(right, text="训练新实验 · 单种子", command=self.train)
        self.train_button.pack(fill="x", pady=6)
        ttk.Button(right, text="停止当前运行 · 保留日志", command=self.stop).pack(fill="x", pady=6)
        self.status = ttk.Label(self, text="就绪；实时3D支持全部控制器，保存GIF为指定基线案例。", wraplength=1000)
        self.status.pack(fill="x", padx=18, pady=12)
        self.load_replay()
        self.after(100, self.tick)

    def python(self):
        exe = Path(sys.executable)
        return exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe

    def launch(self, args, label):
        if self.process:
            messagebox.showinfo("正在运行", "请先停止当前运行，日志与部分结果会保留。")
            return
        folder = ROOT/"outputs/continuous-student"
        folder.mkdir(parents=True, exist_ok=True)
        self.log = folder/(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".log")
        self.stream = self.log.open("w", encoding="utf-8")
        self.process = subprocess.Popen([str(self.python()), str(ROOT/"continuous.py"), *args], stdout=self.stream, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.train_button.configure(state="disabled")
        self.status.configure(text=f"{label}；本次日志：{self.log.name}")

    def live(self):
        if self.process:
            return
        self.pending = None
        self.launch(["--demo", str(self.folder), "--policy", self.policy.get(), "--split", self.split.get(), "--case", self.case.get()], "实时3D运行")

    def train(self):
        if self.process:
            return
        try:
            steps, seed = int(self.steps.get()), int(self.seed.get())
            if steps <= 0 or steps > 1048576 or steps % 1024 or seed < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("参数", "步数为1024的正整数倍，最大1048576；种子为非负整数。")
            return
        folder = ROOT/"outputs/continuous-student"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.pending = folder
        self.pending_seed = seed
        self.launch(["--kind", "arm", "--output", str(folder), "--steps", str(steps), "--seeds", str(seed)], "PPO训练中")

    def load_replay(self):
        self.playing = False
        policy = self.policy.get()
        path = self.folder/"demos"/(policy+".gif")
        self.frames = []
        self.frame = 0
        if path.exists():
            with Image.open(path) as im:
                for i in range(im.n_frames):
                    im.seek(i)
                    self.frames.append(ImageTk.PhotoImage(im.convert("RGB")))
            self.picture.configure(image=self.frames[0], text="")
        else:
            self.picture.configure(image="", text="该控制器没有保存GIF。点击实时3D运行所选域与案例。")
        summary = self.folder/"summary.json"
        self.metrics.configure(state="normal")
        self.metrics.delete("1.0", "end")
        if summary.exists():
            record = json.loads(summary.read_text(encoding="utf-8"))
            values = record["aggregate"].get(policy)
            self.metrics.insert("end", f"固定基线 · ID 24案例汇总\n{json.dumps(values, ensure_ascii=False, indent=2)}\n")
            self.metrics.insert("end", "保存GIF对应ID案例0。实时3D使用右侧域与案例。惯性/伺服已建模，碰撞操作按后续关卡验收。")
        self.metrics.configure(state="disabled")

    def toggle(self):
        self.playing = not self.playing

    def tick(self):
        if self.playing and self.frames:
            self.frame = (self.frame+1) % len(self.frames)
            self.picture.configure(image=self.frames[self.frame])
        if self.process:
            code = self.process.poll()
            if code is None:
                try:
                    lines = self.log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if lines:
                        self.status.configure(text=f"运行中：{lines[-1]}")
                except OSError:
                    pass
            else:
                self.stream.close()
                self.process = self.stream = None
                self.train_button.configure(state="normal")
                if code == 0 and self.pending:
                    self.folder = self.pending
                    self.policy.set("ppo_seed_"+str(self.pending_seed))
                    self.load_replay()
                self.status.configure(text=f"运行结束，退出码{code}；日志：{self.log.name}")
        self.after(100, self.tick)

    def stop(self):
        if self.process:
            self.process.terminate()
        self.playing = False

    def close(self):
        self.stop()
        if self.process:
            self.process.wait(timeout=5)
        if self.stream:
            self.stream.close()
        self.destroy()


if __name__ == "__main__":
    ContinuousPanel().mainloop()

"""Interactive single-transition teaching and asynchronous new-run training."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import ImageTk
from tabular_lab import ACTION_NAMES, ArmMDP, Config, draw_arm, td_target, value_iteration

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/"outputs/baseline-20261013"


class Panel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("强化学习与安全执行 · 项目05")
        self.geometry("1120x930")
        self.minsize(1000, 880)
        self.cfg = Config()
        self.env = ArmMDP(self.cfg)
        self.rng = np.random.default_rng(7)
        self.q = np.zeros((self.env.failure_state+1, 5))
        self.running = False
        self.process = None
        self.logfile = None
        self.run_dir = BASELINE
        self.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(self)
        style.configure("TLabel", font=("Microsoft YaHei UI", 11))
        style.configure("TButton", font=("Microsoft YaHei UI", 11), padding=(10, 8))
        ttk.Label(self, text="观察 → 选择动作 → 奖励 → 更新 Q → 再试", font=("Microsoft YaHei UI", 18, "bold")).pack(anchor="w", padx=20, pady=(16, 8))
        ttk.Label(self, text="固定目标 · 离散运动学二连杆 · 安全范围：关节网格限位", foreground="#134e4a").pack(anchor="w", padx=20)
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=20, pady=12)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16))
        self.picture = ttk.Label(left)
        self.picture.pack(fill="x")
        self.values = ttk.Treeview(left, columns=("action", "before", "after"), show="headings", height=5)
        for key, label in (("action", "动作"), ("before", "Q 更新前"), ("after", "Q 更新后")):
            self.values.heading(key, text=label)
            self.values.column(key, width=150)
        self.values.pack(fill="x", pady=8)
        self.detail = tk.Text(left, height=5, font=("Microsoft YaHei UI", 11), wrap="word", bg="#f0fdfa", fg="#134e4a")
        self.detail.pack(fill="both", expand=True)
        right = ttk.Frame(body, width=310)
        right.pack(side="right", fill="y")
        self.mode = tk.StringVar(value="已训练 Q · 保存模型")
        ttk.Label(right, text="控制策略").pack(anchor="w")
        box = ttk.Combobox(right, textvariable=self.mode, values=["已训练 Q · 保存模型", "随机动作", "未训练 Q", "已知模型最优策略"], state="readonly", width=27)
        box.pack(fill="x", pady=(4, 8))
        box.bind("<<ComboboxSelected>>", lambda _: self.load_policy())
        self.filtered = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="关节限位投影", variable=self.filtered, command=self.reset).pack(anchor="w", pady=8)
        ttk.Label(right, text="起始状态编号（0—440，终点除外）").pack(anchor="w")
        self.start = tk.StringVar(value="210")
        ttk.Entry(right, textvariable=self.start).pack(fill="x", pady=(4, 8))
        for label, callback in (("重置到起点", self.reset), ("执行一步 · 策略保持冻结", self.step),
                                ("学习一步 · 探索并更新 Q", lambda: self.step(learn=True)),
                                ("播放 / 暂停", self.toggle), ("载入公开基线", self.load_baseline),
                                ("打开知识卡与操作课", lambda: os.startfile(ROOT/"01-知识卡与二连杆实操.md")),
                                ("查看训练曲线", lambda: os.startfile(self.run_dir/"learning-curve.png")),
                                ("打开实验结果目录", lambda: os.startfile(self.run_dir))):
            ttk.Button(right, text=label, command=callback).pack(fill="x", pady=(0, 6))
        ttk.Label(right, text="新实验：回合数 / 训练种子").pack(anchor="w", pady=(4, 0))
        fields = ttk.Frame(right)
        fields.pack(fill="x", pady=(4, 8))
        self.episodes, self.seed = tk.StringVar(value="6000"), tk.StringVar(value="5")
        ttk.Entry(fields, textvariable=self.episodes, width=12).pack(side="left")
        ttk.Entry(fields, textvariable=self.seed, width=10).pack(side="left", padx=8)
        self.train_button = ttk.Button(right, text="训练新实验 · 单种子", command=self.train)
        self.train_button.pack(fill="x", pady=(0, 6))
        ttk.Button(right, text="停止本面板启动的训练", command=self.stop).pack(fill="x")
        self.status = ttk.Label(self, text="就绪。每一步的动作、奖励和 Q 更新显示在左侧。", wraplength=1080)
        self.status.pack(fill="x", padx=20, pady=(0, 12))
        self.load_policy()
        self.after(150, self.tick)

    def load_baseline(self):
        self.run_dir = BASELINE
        self.mode.set("已训练 Q · 保存模型")
        self.load_policy()

    def load_policy(self):
        if "已训练" in self.mode.get():
            path = self.run_dir/"q-seed-0.npz"
            if not path.exists():
                choices = sorted(self.run_dir.glob("q-seed-*.npz"))
                path = choices[0] if choices else path
            if path.exists():
                with np.load(path) as model:
                    self.q = model["q"].copy()
                self.status.configure(text=f"已加载 {path.name}；单步学习只修改内存副本。")
            else:
                self.q = np.zeros_like(self.q)
                self.status.configure(text="基线文件缺失：当前使用零表。请训练新实验。")
        elif "最优" in self.mode.get():
            self.q, _ = value_iteration(ArmMDP(self.cfg))
        else:
            self.q = np.zeros_like(self.q)
        self.reset()

    def reset(self):
        self.running = False
        try:
            start = int(self.start.get())
            env = ArmMDP(self.cfg, self.filtered.get())
            env.reset(start)
        except ValueError as exc:
            messagebox.showerror("起点输入", f"{exc}\n请使用0—440内的非终点编号，例如0。")
            return
        self.env = env
        self.show({"q": env.q[start].tolist(), "step": 0, "action": 4,
                   "error": float(env.error[start]), "intervention": False, "violation": False}, self.q[start], self.q[start], "重置完成：预测哪一个动作最有利，再点执行一步。")

    def show(self, row, before, after, detail):
        captions = {"已训练 Q · 保存模型": "Learned Q (loaded seed)", "随机动作": "Random actions",
                    "未训练 Q": "Untrained Q", "已知模型最优策略": "Known-model optimal policy"}
        self.photo = ImageTk.PhotoImage(draw_arm(self.cfg, row, captions[self.mode.get()]))
        self.picture.configure(image=self.photo)
        self.values.delete(*self.values.get_children())
        for name, old, new in zip(ACTION_NAMES, before, after):
            self.values.insert("", "end", values=(name, f"{old:.5f}", f"{new:.5f}"))
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", detail)
        self.detail.configure(state="disabled")

    def step(self, learn=False):
        if self.env.terminal[self.env.state] or self.env.steps >= self.cfg.max_steps:
            self.running = False
            self.status.configure(text="本回合结束。重置后再开始；成功、越界和截断分别显示。")
            return
        state = self.env.state
        before = self.q[state].copy()
        exploring = self.mode.get() == "随机动作" or (learn and self.rng.random() < 0.3)
        action = int(self.rng.integers(5)) if exploring else int(np.argmax(before))
        nxt, reward, terminated, truncated, info = self.env.step(action)
        target = td_target(reward, self.q[nxt], terminated, self.cfg.gamma)
        if learn:
            self.q[state, action] += self.cfg.alpha*(target-before[action])
        row = {"step": self.env.steps, "action": action, **info}
        self.show(row, before, self.q[state],
                  f"状态 {state} → {nxt}；动作 {ACTION_NAMES[action]}；奖励 {reward:.5f}\n"
                  f"TD目标 = r + 0.97 × (1−terminated) × maxQ(s′) = {target:.5f}\n"
                  f"本步学习={learn}；terminated={terminated}；truncated={truncated}；成功={info['success']}\n"
                  f"投影干预={info['intervention']}；越界={info['violation']}；误差={info['error']:.4f} m")
        if terminated or truncated:
            self.running = False

    def toggle(self):
        self.running = not self.running

    def train(self):
        if self.process:
            return
        try:
            episodes, seed = int(self.episodes.get()), int(self.seed.get())
            if not 1 <= episodes <= 100000 or seed < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("训练输入", "回合数范围1—100000，种子为非负整数。")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.pending_dir = ROOT/"outputs/student"/stamp
        self.pending_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.pending_dir.parent/f"{stamp}.log"
        self.logfile = log_path.open("w", encoding="utf-8")
        exe = Path(sys.executable)
        if exe.name.lower() == "pythonw.exe":
            exe = exe.with_name("python.exe")
        self.process = subprocess.Popen([str(exe), str(ROOT/"tabular_lab.py"), "--output", str(self.pending_dir),
                                         "--episodes", str(episodes), "--seeds", str(seed)],
                                        stdout=self.logfile, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.train_button.configure(state="disabled")
        self.status.configure(text=f"训练进行中，输出保留到 {self.pending_dir.name}。面板仍可单步操作。")

    def tick(self):
        if self.running:
            self.step()
        if self.process:
            code = self.process.poll()
            if code is not None:
                self.logfile.close()
                self.logfile = None
                self.process = None
                self.train_button.configure(state="normal")
                if code == 0:
                    self.run_dir = self.pending_dir
                    self.mode.set("已训练 Q · 保存模型")
                    self.load_policy()
                    self.status.configure(text=f"训练完成；已载入新表。结果路径：{self.run_dir}")
                else:
                    self.status.configure(text=f"训练退出码 {code}；日志与部分输出保留。")
        self.after(150, self.tick)

    def stop(self):
        if self.process:
            self.process.terminate()
        self.running = False

    def close(self):
        self.stop()
        if self.process:
            self.process.wait(timeout=5)
        if self.logfile:
            self.logfile.close()
        self.destroy()


if __name__ == "__main__":
    Panel().mainloop()

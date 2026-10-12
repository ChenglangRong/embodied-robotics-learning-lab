"""Native learning panel: replay results, edit parameters and run real physics."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageSequence

ROOT = Path(__file__).resolve().parent
ROBOTS = {
    "panda": ("01 · Panda 三维抓放", "观察：接近 → 下降 → 合拢 → 抬升 → 搬运 → 放开。\n橙色方块靠夹爪接触与摩擦运动；绿色区域是放置目标。", "close_ctrl", 0, 255, 0),
    "go2": ("02 · Go2 四足前进", "观察：对角腿交替摆动、足端离地、身体向前移动。\n修改 PD 增益，比较目标步态与实际运动方向。", "kp", 40, 160, 100),
    "h1": ("03 · H1 平衡下蹲", "观察：自由浮动身体、双足支撑、下蹲、抬臂、站回。\n质心反馈驱动踝关节校正；增益过小会失稳。", "com_gain", 1, 8, 5),
    "allegro": ("04 · Allegro 多指托抓", "观察：手指合拢、支撑台下降、腕部抬升、球保持接触。\n腕部为可控夹具，球与 16 个关节参与物理仿真。", "wrist_tilt", 0, 1, 0),
}
FAILURE = {
    "panda": {"close_ctrl": 255}, "go2": {"kp": 55}, "h1": {"com_gain": 1.5},
    "allegro": {"wrist_quat": "1 0 0 0", "object_pos": [.045, 0, .175], "radius": .035,
                "closed": [0, 1.15, 1.35, .8] * 3 + [1.1, .6, .9, .7]},
}


class Panel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("机器人进阶实验室 · 项目03")
        self.geometry("1040x860")
        self.configure(bg="#eaf0f5")
        self.process = None
        self.study_running = False
        self.frames = []
        self.frame_index = 0
        self.playing = True
        self.after_id = None
        ttk.Label(self, text="四类机器人 · 先观察结果，再修改代码与参数", font=("Microsoft YaHei UI", 18, "bold")).pack(pady=(18, 8))
        ttk.Label(self, text="回放展示已保存的实验；运行按钮启动当前模型的实时 MuJoCo 物理计算。").pack()
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="x", padx=22, pady=12)
        self.keys = list(ROBOTS)
        for kind, (title, *_rest) in ROBOTS.items():
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text=title)
        self.tabs.bind("<<NotebookTabChanged>>", lambda event: self.select())
        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, padx=22)
        self.image_label = ttk.Label(content)
        self.image_label.grid(row=0, column=0, rowspan=11, padx=(0, 18), sticky="n")
        self.info = ttk.Label(content, wraplength=300, font=("Microsoft YaHei UI", 10), justify="left")
        self.info.grid(row=0, column=1, sticky="nw")
        self.param_label = ttk.Label(content)
        self.param_label.grid(row=1, column=1, sticky="w", pady=(12, 0))
        self.param = tk.DoubleVar(value=0)
        self.scale = ttk.Scale(content, variable=self.param, command=lambda _: self.show_param())
        self.scale.grid(row=2, column=1, sticky="ew")
        ttk.Button(content, text="运行物理实验（3D 窗口）", command=self.start).grid(row=3, column=1, sticky="ew", pady=8)
        ttk.Button(content, text="运行失败对照实验", command=lambda: self.start(failure=True)).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Button(content, text="暂停 / 继续回放", command=self.toggle).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Button(content, text="打开实操教程", command=lambda: subprocess.Popen(["notepad.exe", str(ROOT / "实操课.md")])).grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Button(content, text="打开控制器代码 lab.py", command=lambda: subprocess.Popen(["notepad.exe", str(ROOT / "lab.py")])).grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Button(content, text="深入训练：知识卡片与迁移", command=lambda: subprocess.Popen(["notepad.exe", str(ROOT / "深入训练与迁移.md")])).grid(row=8, column=1, sticky="ew", pady=4)
        ttk.Button(content, text="运行本平台单因素组（自动出图）", command=self.start_study).grid(row=9, column=1, sticky="ew", pady=4)
        ttk.Button(content, text="打开深入训练报告", command=self.open_study_report).grid(row=10, column=1, sticky="ew", pady=4)
        self.result = tk.Text(self, height=6, wrap="word", font=("Consolas", 10), bg="#111c2a", fg="#e5eef7", relief="flat")
        self.result.pack(fill="x", padx=22, pady=(14, 8))
        self.status = ttk.Label(self, text="就绪 · 先完成 Panda，再按标签顺序推进。")
        self.status.pack(anchor="w", padx=22, pady=(0, 14))
        self.select()
        self.after(100, self.animate)

    def kind(self):
        return self.keys[self.tabs.index(self.tabs.select())]

    def show_param(self):
        kind = self.kind()
        self.param_label.configure(text=f"实验参数 {ROBOTS[kind][2]} = {self.param.get():.3f}")

    def select(self):
        if not hasattr(self, "info"):
            return
        kind = self.kind()
        _, desc, _, low, high, default = ROBOTS[kind]
        self.info.configure(text=desc)
        self.scale.configure(from_=low, to=high)
        self.param.set(default)
        self.show_param()
        self.load_result(ROOT / "outputs/baseline" / kind)

    def load_result(self, path):
        self.frames = []
        gif = path / "demo.gif"
        if gif.exists():
            with Image.open(gif) as im:
                self.frames = [ImageTk.PhotoImage(frame.convert("RGB")) for frame in ImageSequence.Iterator(im)]
            self.frame_index = 0
            self.image_label.configure(image=self.frames[0])
        result_file = path / "summary.json"
        if result_file.exists():
            r = json.loads(result_file.read_text(encoding="utf-8"))
            text = ("验收通过" if r["passed"] else "验收失败：请检查任务指标") + f" · {r['task']}\n"
            text += json.dumps(r["metrics"], ensure_ascii=False, indent=2)
        else:
            text = "运行物理实验后生成结果。"
        self.result.delete("1.0", "end")
        self.result.insert("1.0", text)

    def toggle(self):
        self.playing = not self.playing

    def animate(self):
        if self.frames and self.playing:
            self.image_label.configure(image=self.frames[self.frame_index])
            self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.after(100, self.animate)

    def start(self, failure=False):
        if self.process and self.process.poll() is None:
            messagebox.showinfo("实验正在运行", "请观看当前 3D 窗口；关闭窗口或等待运行结束后再启动。")
            return
        kind = self.kind()
        self.study_running = False
        opt = FAILURE[kind] if failure else {ROBOTS[kind][2]: self.param.get()}
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.last_output = ROOT / "outputs/user-runs" / f"{stamp}-{kind}"
        self.last_output.mkdir(parents=True)
        self.log = (self.last_output / "console.log").open("w", encoding="utf-8")
        py = Path(sys.executable)
        if py.name.lower() == "pythonw.exe":
            py = py.with_name("python.exe")
        self.process = subprocess.Popen([str(py), str(ROOT / "lab.py"), kind, "--viewer", "--render", "--output", str(self.last_output), "--options", json.dumps(opt)],
                                        stdout=self.log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.status.configure(text=f"运行中：{kind} · 在 MuJoCo 窗口拖动视角、滚轮缩放；完成后保留画面 15 秒。")
        self.after(500, self.poll)

    def open_study_report(self):
        report = getattr(self, "study_report", ROOT / "outputs/deep-practice/mentor-20261012-v2/report.html")
        if report.exists():
            webbrowser.open(report.resolve().as_uri())
        else:
            messagebox.showinfo("实验报告", "点击运行本平台单因素组，结束后生成报告。")

    def start_study(self):
        if self.process and self.process.poll() is None:
            messagebox.showinfo("实验正在运行", "等待当前实验结束后启动对照组。")
            return
        kind = self.kind()
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.last_output = ROOT / "outputs/deep-practice" / f"{stamp}-{kind}"
        log_dir = ROOT / "outputs/user-runs" / f"{stamp}-study-{kind}"
        log_dir.mkdir(parents=True)
        self.log = (log_dir / "console.log").open("w", encoding="utf-8")
        py = Path(sys.executable)
        if py.name.lower() == "pythonw.exe":
            py = py.with_name("python.exe")
        self.study_running = True
        self.study_report = self.last_output / "report.html"
        self.process = subprocess.Popen([str(py), str(ROOT / "deep_practice.py"), "--robot", kind, "--output", str(self.last_output)],
                                        stdout=self.log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.status.configure(text=f"计算 {kind} 三个单因素对照；完成后打开曲线报告。3D 实时演示使用上方运行按钮。")
        self.after(500, self.poll)

    def poll(self):
        if self.process.poll() is None:
            self.after(500, self.poll)
            return
        self.log.close()
        if self.process.returncode != 0:
            messagebox.showerror("运行出错", Path(self.log.name).read_text(encoding="utf-8"))
            self.status.configure(text="实验出错：日志已保留，查看弹窗中的错误。")
            return
        if self.study_running:
            self.result.delete("1.0", "end")
            self.result.insert("1.0", f"单因素组已完成：{self.last_output}\n打开报告查看三条对照曲线与原始诊断数据。")
            self.status.configure(text=f"深入训练报告：{self.study_report}")
            self.open_study_report()
            return
        self.load_result(self.last_output)
        self.status.configure(text=f"本次结果：{self.last_output}")


if __name__ == "__main__":
    Panel().mainloop()

"""Five-platform lesson hub, with asynchronous training and real MuJoCo demos."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import webbrowser
from PIL import Image,ImageTk

ROOT=Path(__file__).resolve().parent
NAMES={"planar":"二连杆", "panda":"Panda机械臂", "go2":"Go2机器狗", "h1":"H1人形", "allegro":"Allegro灵巧手"}
DESCRIPTIONS={"planar":"6维观测→2维关节速度；BC、DAgger与1 mm到达评测。",
              "panda":"学习7关节参考与夹爪开合，完成抓取、搬运和放置；读取物体相对末端位姿。",
              "go2":"学习12关节步态参考，由固定PD执行；记录前进、倾斜、足地接触和滑动。",
              "h1":"学习19关节参考，包含平衡修正、下蹲和抬臂；固定PD与重力偏置补偿保留。",
              "allegro":"学习16指关节参考，观察多指/掌面受力与物体相对位姿；腕部夹具与支撑撤去由环境规定。"}


def python_executable():
    path=Path(sys.executable)
    return path.with_name("python.exe") if path.name.lower()=="pythonw.exe" else path


class PlatformPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("五平台模仿学习 · 项目04")
        self.geometry("1120x850")
        self.minsize(920,720)
        self.process=None
        self.animation=False
        self.frames=[]
        self.index=0
        self.last_demos={}
        self.protocol("WM_DELETE_WINDOW",self.close)
        style=ttk.Style(self)
        style.configure("TButton",padding=(10,7))
        style.configure("TLabel",font=("Microsoft YaHei UI",11))
        ttk.Label(self,text="五个平台 · 示范 → 训练 → 独立执行 → 扰动验证",font=("Microsoft YaHei UI",18,"bold")).pack(anchor="w",padx=18,pady=(16,8))
        self.status=ttk.Label(self,text="就绪：选择平台。快速版为1训练种子，正式版为5训练种子。",wraplength=1050)
        self.status.pack(side="bottom",anchor="w",padx=18,pady=(4,12))
        self.progress=ttk.Progressbar(self,mode="indeterminate")
        self.progress.pack(side="bottom",fill="x",padx=18)
        self.tabs=ttk.Notebook(self)
        self.tabs.pack(fill="x",padx=18,pady=(0,8))
        for name in NAMES.values():self.tabs.add(ttk.Frame(self.tabs),text=name)
        self.tabs.bind("<<NotebookTabChanged>>",lambda _:self.load_selected())
        self.description=ttk.Label(self,text="",wraplength=1040)
        self.description.pack(anchor="w",padx=18,pady=(0,10))
        body=ttk.Frame(self); body.pack(fill="both",expand=True,padx=18,pady=(0,8))
        left=ttk.Frame(body); left.pack(side="left",fill="both",expand=True,padx=(0,18))
        self.picture=ttk.Label(left,text="训练完成后载入真实动力学回放",anchor="center")
        self.picture.pack(fill="both",expand=True)
        self.caption=ttk.Label(left,text="回放来源待加载",wraplength=650)
        self.caption.pack(anchor="w",pady=8)
        self.metrics=tk.Text(left,height=7,wrap="word",font=("Microsoft YaHei UI",11),bg="#eef6f7",fg="#172b3a")
        self.metrics.pack(fill="x")
        right_shell=ttk.Frame(body,width=310);right_shell.pack(side="right",fill="y")
        controls=tk.Canvas(right_shell,width=310,highlightthickness=0)
        controls.pack(side="left",fill="both",expand=True)
        control_scroll=ttk.Scrollbar(right_shell,orient="vertical",command=controls.yview)
        control_scroll.pack(side="right",fill="y")
        controls.configure(yscrollcommand=control_scroll.set)
        right=ttk.Frame(controls)
        control_window=controls.create_window((0,0),window=right,anchor="nw")
        right.bind("<Configure>",lambda _:controls.configure(scrollregion=controls.bbox("all")))
        controls.bind("<Configure>",lambda event:controls.itemconfigure(control_window,width=event.width))
        def wheel(event):
            widget=self.winfo_containing(event.x_root,event.y_root)
            while widget is not None:
                if widget is right_shell:
                    controls.yview_scroll(-1 if event.delta>0 else 1,"units")
                    return
                widget=widget.master
        self.bind("<MouseWheel>",wheel,add="+")
        ttk.Label(right,text="控制器").pack(anchor="w")
        self.policy=tk.StringVar(value="expert")
        ttk.Combobox(right,textvariable=self.policy,values=["expert","bc","phase_bc","frozen_state","hold"],state="readonly").pack(fill="x",pady=(4,8))
        ttk.Label(right,text="测试域 / 固定案例编号").pack(anchor="w")
        fields=ttk.Frame(right);fields.pack(fill="x",pady=(4,8))
        self.split=tk.StringVar(value="test"); self.case=tk.StringVar(value="0")
        ttk.Combobox(fields,textvariable=self.split,values=["test","ood"],state="readonly",width=9).pack(side="left")
        self.case_box=ttk.Combobox(fields,textvariable=self.case,values=["0","1"],state="readonly",width=8)
        self.case_box.pack(side="left",padx=8)
        ttk.Button(right,text="运行所选控制器 · 实时3D",command=self.live_demo).pack(fill="x",pady=(0,8))
        ttk.Button(right,text="载入所选控制器回放",command=self.load_replay).pack(fill="x",pady=(0,8))
        ttk.Button(right,text="回放：播放 / 暂停",command=self.toggle).pack(fill="x",pady=(0,12))
        ttk.Label(right,text="训练更新数（空＝默认）").pack(anchor="w")
        self.updates=tk.StringVar(value="")
        ttk.Entry(right,textvariable=self.updates).pack(fill="x",pady=(4,8))
        self.training_buttons=[]
        for label,mode in (("本平台快速实验 · 1种子","quick"),("本平台正式实验 · 5种子","full")):
            button=ttk.Button(right,text=label,command=lambda m=mode:self.train(m))
            button.pack(fill="x",pady=(0,8));self.training_buttons.append(button)
        ttk.Button(right,text="载入已测四平台样例",command=self.load_sample).pack(fill="x",pady=(0,8))
        ttk.Button(right,text="打开结果报告",command=self.open_report).pack(fill="x",pady=(0,8))
        ttk.Button(right,text="打开五平台实操课",command=lambda:os.startfile(ROOT/"03-五平台模仿学习实操课.md")).pack(fill="x",pady=(0,8))
        ttk.Button(right,text="打开二连杆训练与单步面板",command=self.open_planar).pack(fill="x",pady=(0,8))
        ttk.Button(right,text="停止当前运行 · 保留部分输出",command=self.stop).pack(fill="x")
        latest=ROOT/"outputs/platform-latest.txt"
        candidate=Path(latest.read_text(encoding="utf-8").strip()) if latest.exists() else None
        self.run_dir=candidate if candidate and (candidate/"summary.json").exists() else ROOT/"outputs/platform-baseline-20261012-r3"
        self.tabs.select(1)
        self.load_selected()
        self.after(100,self.tick)

    def kind(self):return list(NAMES)[self.tabs.index("current")]

    def set_metrics(self,text):
        self.metrics.configure(state="normal");self.metrics.delete("1.0","end");self.metrics.insert("1.0",text);self.metrics.configure(state="disabled")

    def load_selected(self):
        if not hasattr(self,"picture"):return
        kind=self.kind()
        self.description.configure(text=DESCRIPTIONS[kind])
        if kind=="planar":
            self.frames=[];self.picture.configure(image="",text="二连杆：点击右侧按钮打开训练与单步面板")
            self.set_metrics("已测二连杆基线：5种子、1 mm。详细结果在原面板中查看。")
            return
        source=self.run_dir/"summary.json"
        if source.exists():
            summary=json.loads(source.read_text(encoding="utf-8"))
            record=summary.get("platforms",{}).get(kind)
            if record:
                text=f"结果：{self.run_dir.name} | {summary['mode']} | 训练种子 {summary['training_seeds']}\n"
                text+=f"obs {record['observation_dim']}维 → action {record['action_dim']}维；PD/位置伺服保留\n"
                for name,row in record["aggregate"].items():text+=f"{name}: ID {row['test']['mean_success_rate']:.1%} / OOD {row['ood']['mean_success_rate']:.1%}\n"
                self.set_metrics(text)
                p=json.loads((self.run_dir/kind/"protocol.json").read_text(encoding="utf-8"))
                self.case_box.configure(values=[str(i) for i in range(len(p["specs"]["test"]))])
                self.case.set("0")
            else:self.set_metrics("此平台的训练结果待生成；点击本平台快速实验。")
        else:self.set_metrics("完整样例待生成；点击本平台快速实验。")
        self.load_replay()

    def load_sample(self):
        self.run_dir=ROOT/"outputs/platform-baseline-20261012-r3"
        self.load_selected()

    def load_replay(self):
        kind=self.kind()
        if kind=="planar":return
        name=self.policy.get()
        path=self.last_demos.get((kind,name)) or self.run_dir/kind/"demos"/name
        source=path/"demo.gif"
        self.frames=[];self.index=0;self.animation=False
        if not source.exists():
            self.picture.configure(image="",text="所选控制器回放待生成；点击实时3D运行并保存新结果。")
            self.caption.configure(text="当前回放：待生成")
            return
        with Image.open(source) as gif:
            for i in range(gif.n_frames):
                gif.seek(i);self.frames.append(ImageTk.PhotoImage(gif.convert("RGB")))
        self.picture.configure(image=self.frames[0],text="")
        self.caption.configure(text=f"实际回放：{kind} / {name} / {path}；按播放开始。")

    def toggle(self):self.animation=not self.animation

    def tick(self):
        if self.animation and self.frames:
            self.index=(self.index+1)%len(self.frames);self.picture.configure(image=self.frames[self.index])
        self.after(100,self.tick)

    def launch(self,args,output,training=False):
        if self.process and self.process.poll() is None:
            self.status.configure(text="当前任务运行中，完成后再启动下一项。");return
        console=ROOT/"outputs/platform-console";console.mkdir(parents=True,exist_ok=True)
        self.log_path=console/(output.name+".log")
        self.log=self.log_path.open("w",encoding="utf-8")
        try:
            self.process=subprocess.Popen([str(python_executable()),"-u",str(ROOT/"platform_learning.py"),*args],cwd=ROOT,
                                          stdout=self.log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        except OSError as error:
            self.log.close();self.status.configure(text=f"启动失败：{error}；核对Python环境后重试。");return
        self.active_output,self.active_training=output,training
        self.active_kind,self.active_policy=self.kind(),self.policy.get()
        self.stopped=False
        for button in self.training_buttons:button.configure(state="disabled")
        self.progress.start(15);self.status.configure(text=f"运行中，日志：{self.log_path}")
        self.after(500,self.poll)

    def train(self,mode):
        kind=self.kind()
        if kind=="planar":self.open_planar();return
        steps=self.updates.get().strip()
        if steps and (not steps.isdigit() or int(steps)<1):
            self.status.configure(text="训练更新数需要正整数，留空使用quick1800/full3600。");return
        output=ROOT/"outputs/platform-runs"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        args=["--robot",kind,"--mode",mode,"--output",str(output)]
        if steps:args.extend(["--steps",steps])
        self.launch(args,output,True)

    def live_demo(self):
        kind=self.kind()
        if kind=="planar":self.open_planar();return
        if not (self.run_dir/kind/"protocol.json").exists():
            self.status.configure(text="先运行本平台快速实验，生成协议与BC权重，再启动3D对照。");return
        output=ROOT/"outputs/platform-runs"/datetime.datetime.now().strftime("demo-%Y%m%d-%H%M%S-%f")
        self.launch(["--robot",kind,"--demo-run",str(self.run_dir),"--policy",self.policy.get(),"--split",self.split.get(),"--case",self.case.get(),"--viewer","--output",str(output)],output)

    def poll(self):
        lines=self.log_path.read_text(encoding="utf-8",errors="replace").splitlines()
        if lines:self.status.configure(text=lines[-1])
        if self.process.poll() is None:self.after(500,self.poll);return
        self.log.close();self.progress.stop()
        for button in self.training_buttons:button.configure(state="normal")
        if self.process.returncode:
            self.status.configure(text=f"{'已停止并保留部分输出' if self.stopped else '运行失败'}；日志{self.log_path}")
            self.set_metrics("\n".join(lines[-10:]));return
        if self.active_training:
            self.run_dir=self.active_output;self.load_selected();self.open_report()
        else:
            self.last_demos[(self.active_kind,self.active_policy)]=self.active_output
            self.tabs.select(list(NAMES).index(self.active_kind))
            self.policy.set(self.active_policy)
            self.load_replay()
            result=json.loads((self.active_output/"summary.json").read_text(encoding="utf-8"))
            self.set_metrics(json.dumps({k:result[k] for k in ("policy","passed","completed","criteria","expert_calls","metrics")},ensure_ascii=False,indent=2))
        self.status.configure(text=f"完成：{self.active_output}")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stopped=True;self.process.terminate();self.status.configure(text="停止当前子进程，保留日志与部分输出；完整验收需重新运行。");return
        self.status.configure(text="当前没有正在运行的任务。")

    def open_report(self):
        path=self.run_dir/"report.html"
        if path.exists():webbrowser.open(path.as_uri())
        else:self.status.configure(text="训练完成后生成完整报告。")

    def open_planar(self):
        subprocess.Popen([str(python_executable()),str(ROOT/"panel.py")],cwd=ROOT,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)

    def close(self):
        if self.process and self.process.poll() is None:
            self.status.configure(text="任务仍在运行；完成后关闭，或点停止保留部分结果。");return
        self.destroy()


if __name__=="__main__":PlatformPanel().mainloop()

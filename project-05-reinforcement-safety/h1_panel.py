"""Desktop H1 push-recovery lesson: physical replay, matched ablations and live runs."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk,messagebox
from PIL import Image,ImageTk
from h1_recovery import CONTROLLERS,SCENARIOS

ROOT=Path(__file__).resolve().parent
BASELINE=ROOT/"outputs/h1-recovery-20261016-r2"
LESSONS=("07-H1扰动恢复实操课.md","08-H1扰动恢复实操.md")


def shown(value,scale=1.,digits=2):return "n/a" if value is None else f"{value*scale:.{digits}f}"


class H1Panel(tk.Tk):
    """The controller and scenario lists follow the frozen module, not a local copy."""

    def __init__(self):
        super().__init__();self.title("H1双足扰动恢复实验");self.geometry("1180x860");self.minsize(1100,820)
        self.process=self.stream=self.pending=None;self.frames=[];self.frame=0;self.playing=False
        self.folder=BASELINE;self.logdir=ROOT/"outputs/h1-student"
        self.protocol("WM_DELETE_WINDOW",self.close)
        style=ttk.Style(self);style.configure("TButton",font=("Microsoft YaHei UI",11),padding=(10,7));style.configure("TLabel",font=("Microsoft YaHei UI",11))
        ttk.Label(self,text="H1：质心误差 → 踝关节参考修正 → 关节PD → 足地接触（双足扰动恢复）",font=("Microsoft YaHei UI",16,"bold")).pack(anchor="w",padx=16,pady=12)
        body=ttk.Frame(self);body.pack(fill="both",expand=True,padx=16)
        left=ttk.Frame(body);left.pack(side="left",fill="both",expand=True,padx=(0,16))
        self.tabs=ttk.Notebook(left);self.tabs.pack(fill="both",expand=True)
        replay=ttk.Frame(self.tabs);metrics=ttk.Frame(self.tabs)
        self.tabs.add(replay,text="真实物理回放");self.tabs.add(metrics,text="案例指标与失败条件")
        self.picture=ttk.Label(replay,anchor="center");self.picture.pack(fill="both",expand=True)
        self.metrics=tk.Text(metrics,wrap="word",font=("Microsoft YaHei UI",11),bg="#f8fafc",fg="#0f172a");self.metrics.pack(fill="both",expand=True)
        right=ttk.Frame(body);right.pack(side="right",fill="y")
        self.mode,self.scenario,self.seed=tk.StringVar(value="com"),tk.StringVar(value="strong_front"),tk.StringVar(value="3000000")
        self.selectors={}
        for label,var,values in (("控制器",self.mode,CONTROLLERS),("场景",self.scenario,SCENARIOS),
                                 ("实时重置种子",self.seed,[str(s) for s in range(3000000,3000005)])):
            ttk.Label(right,text=label).pack(anchor="w",pady=(8,0))
            box=ttk.Combobox(right,textvariable=var,values=values,state="readonly");box.pack(fill="x",pady=6);self.selectors[label]=box
        self.buttons={}
        for label,callback in (("实时3D · 当前选择",self.live),("载入保存回放与指标",self.replay),("回放播放 / 暂停",self.toggle),
                               ("运行新实验 · 五种子全对照",self.experiment),
                               ("打开知识与编程操作课（07-H1扰动恢复实操课.md）",self.lesson),
                               ("打开结果与失败轨迹",self.results),("停止当前运行 · 保留日志",self.stop)):
            button=ttk.Button(right,text=label,command=callback);button.pack(fill="x",pady=6);self.buttons[label]=button
        self.status=ttk.Label(self,text="保存回放来自冻结实验；实时3D使用右侧控制器/场景/种子；新实验与日志写入outputs/h1-student。",wraplength=1120)
        self.status.pack(fill="x",padx=16,pady=12)
        self.replay();self.after(100,self.tick)

    def live_args(self):return ["--demo","--controller",self.mode.get(),"--scenario",self.scenario.get(),"--seed",self.seed.get()]
    def experiment_args(self,folder):return ["--output",str(folder)]
    def lesson_path(self):return next((ROOT/name for name in LESSONS if (ROOT/name).exists()),None)
    def results_path(self):return self.folder if self.folder.exists() else None

    def replay(self):
        self.playing=False;self.frames=[];self.frame=0;name=f"{self.mode.get()}-{self.scenario.get()}"
        gif=self.folder/"demos"/(name+".gif")
        if gif.exists():
            with Image.open(gif) as im:
                for i in range(im.n_frames):im.seek(i);self.frames.append(ImageTk.PhotoImage(im.convert("RGB")))
            self.picture.configure(image=self.frames[0],text="")
        else:
            self.picture.configure(image="",text=f"{name}.gif 未生成；保存回放覆盖fixed/com的strong_front与push_left，实时3D支持全部组合。")
        self.metrics.configure(state="normal");self.metrics.delete("1.0","end")
        path=self.folder/"cases.json"
        if path.exists():
            cases=json.loads(path.read_text(encoding="utf-8"))
            selected=[c for c in cases if c["controller"]==self.mode.get() and c["scenario"]==self.scenario.get()]
            if selected:
                self.metrics.insert("end",f"冻结结果 · {self.mode.get()} / {self.scenario.get()}：同场景{sum(c['success'] for c in selected)}/{len(selected)}通过\n")
                for c in selected:
                    self.metrics.insert("end",f"种子{c['seed']}：success={c['success']} 质心误差峰值={shown(c['peak_com_error_m'],100)}cm "
                                               f"推后峰值位移={shown(c['push_peak_com_error_m'],100)}cm 侧向峰值={shown(c['max_lateral_error_m'],100)}cm "
                                               f"恢复时间={shown(c['recovery_time_s'])}s 最小高度={shown(c['min_height_m'],1.,3)}m "
                                               f"饱和占比={shown(c['saturation_fraction'],100.,1)}% 失败条件={[k for k,v in c['criteria'].items() if not v]}\n")
        else:
            self.metrics.insert("end",f"冻结结果尚未生成：{self.folder}\n")
        self.metrics.insert("end","保存回放固定首种子与四个组合；实时与新实验用子进程调用h1_recovery.py；解析工程基线，RL训练次数0。")
        self.metrics.configure(state="disabled")

    def launch(self,args,newfolder=None):
        if self.process:return
        self.pending=newfolder;self.logdir.mkdir(parents=True,exist_ok=True)
        self.log=self.logdir/(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".log")
        self.stream=self.log.open("w",encoding="utf-8")
        exe=Path(sys.executable);exe=exe.with_name("python.exe") if exe.name.lower()=="pythonw.exe" else exe
        self.process=subprocess.Popen([str(exe),str(ROOT/"h1_recovery.py"),*args],stdout=self.stream,stderr=subprocess.STDOUT,
                                      creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        self.status.configure(text=f"运行中；日志{self.log.name}")

    def live(self):
        if self.process:return
        self.launch(self.live_args())
        self.status.configure(text=f"实时3D：{self.mode.get()} / {self.scenario.get()} 种子{self.seed.get()}；日志{self.log.name}")

    def experiment(self):
        if self.process:return
        folder=self.logdir/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.launch(self.experiment_args(folder),folder)
        self.status.configure(text=f"新实验四控制器×五场景×五种子写入{folder.name}；日志{self.log.name}")

    def lesson(self):
        path=self.lesson_path()
        if path:os.startfile(path)
        else:messagebox.showinfo("知识与编程操作课",f"课程文件尚未生成：{'或'.join(LESSONS)}")

    def results(self):
        path=self.results_path()
        if path:os.startfile(path)
        else:messagebox.showinfo("结果与失败轨迹",f"冻结实验目录尚未生成：{self.folder}")

    def toggle(self):self.playing=not self.playing

    def tick(self):
        if self.playing and self.frames:self.frame=(self.frame+1)%len(self.frames);self.picture.configure(image=self.frames[self.frame])
        if self.process:
            code=self.process.poll()
            if code is None:
                lines=self.log.read_text(encoding="utf-8",errors="replace").splitlines()
                if lines:self.status.configure(text="运行中："+lines[-1])
            else:
                self.stream.close();self.process=self.stream=None
                if code==0 and self.pending:
                    self.folder=self.pending;self.pending=None;self.replay()
                self.status.configure(text=f"运行结束，退出码{code}；日志{self.log.name}")
        self.after(100,self.tick)

    def stop(self):
        if self.process:self.process.terminate()
        self.playing=False

    def close(self):
        self.stop()
        if self.process:self.process.wait(timeout=5)
        if self.stream:self.stream.close()
        self.destroy()


if __name__=="__main__":H1Panel().mainloop()

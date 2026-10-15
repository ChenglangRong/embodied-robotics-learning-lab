"""Desktop Go2 command/response lesson: physical replay, curves and live ablations."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk,messagebox
from PIL import Image,ImageTk
from go2_plot import curve

ROOT=Path(__file__).resolve().parent
BASELINE=ROOT/"outputs/go2-feedback-20261015"
CONTROLLERS=("fixed","feedforward","feedback","wrong_sign")
SCENARIOS=("slow","fast","left","right","step_stop")


class Go2Panel(tk.Tk):
    def __init__(self):
        super().__init__();self.title("Go2速度与转向反馈实验");self.geometry("1140x840");self.minsize(1080,800)
        self.process=self.stream=None;self.frames=[];self.frame=0;self.playing=False;self.folder=BASELINE
        self.protocol("WM_DELETE_WINDOW",self.close)
        style=ttk.Style(self);style.configure("TButton",font=("Microsoft YaHei UI",11),padding=(10,7));style.configure("TLabel",font=("Microsoft YaHei UI",11))
        ttk.Label(self,text="Go2：速度命令 → 步幅反馈 → 腿部IK → 关节PD → 足地接触",font=("Microsoft YaHei UI",16,"bold")).pack(anchor="w",padx=16,pady=12)
        body=ttk.Frame(self);body.pack(fill="both",expand=True,padx=16)
        left=ttk.Frame(body);left.pack(side="left",fill="both",expand=True,padx=(0,16))
        self.tabs=ttk.Notebook(left);self.tabs.pack(fill="both",expand=True)
        replay=ttk.Frame(self.tabs);plot=ttk.Frame(self.tabs);self.tabs.add(replay,text="真实物理回放");self.tabs.add(plot,text="命令 原始值 0.5秒均值")
        self.picture=ttk.Label(replay,anchor="center");self.picture.pack(fill="both",expand=True)
        self.plot=ttk.Label(plot,anchor="center");self.plot.pack(fill="both",expand=True)
        self.metrics=tk.Text(left,height=8,wrap="word",font=("Microsoft YaHei UI",11),bg="#f8fafc",fg="#0f172a");self.metrics.pack(fill="x",pady=8)
        right=ttk.Frame(body);right.pack(side="right",fill="y")
        self.mode,self.scenario,self.seed=tk.StringVar(value="feedback"),tk.StringVar(value="left"),tk.StringVar(value="1000000")
        for label,var,values in (("控制器",self.mode,CONTROLLERS),("场景 slow/fast/left/right/step_stop",self.scenario,SCENARIOS),("实时重置种子",self.seed,[str(s) for s in range(1000000,1000005)])):
            ttk.Label(right,text=label).pack(anchor="w",pady=(8,0));ttk.Combobox(right,textvariable=var,values=values,state="readonly").pack(fill="x",pady=6)
        ttk.Label(right,text="实时/新实验 Kv 与 Kω").pack(anchor="w",pady=(8,0));line=ttk.Frame(right);line.pack(fill="x",pady=6)
        self.kv,self.kw=tk.StringVar(value="2"),tk.StringVar(value="2")
        ttk.Entry(line,textvariable=self.kv,width=10).pack(side="left");ttk.Entry(line,textvariable=self.kw,width=10).pack(side="left",padx=8)
        for label,callback in (("实时3D · 当前选择",self.live),("载入保存回放与曲线",self.replay),("回放播放 / 暂停",self.toggle),
                               ("运行新实验 · 五种子全对照",self.experiment),("停止当前运行 · 保留日志",self.stop),
                               ("打开知识与编程操作课",lambda:os.startfile(ROOT/"06-Go2速度与转向反馈实操.md")),("打开结果与失败轨迹",lambda:os.startfile(self.folder))):
            ttk.Button(right,text=label,command=callback).pack(fill="x",pady=6)
        self.status=ttk.Label(self,text="保存回放与曲线固定种子1000000；实时3D采用右侧种子和增益。",wraplength=1090);self.status.pack(fill="x",padx=16,pady=12)
        self.replay();self.after(100,self.tick)

    def replay(self):
        self.playing=False;self.frames=[];self.frame=0;name=f"{self.mode.get()}-{self.scenario.get()}"
        gif=self.folder/"demos"/(name+".gif");csv=self.folder/"demos"/(name+".csv")
        if gif.exists():
            with Image.open(gif) as im:
                for i in range(im.n_frames):im.seek(i);self.frames.append(ImageTk.PhotoImage(im.convert("RGB")))
            self.picture.configure(image=self.frames[0],text="")
        else:self.picture.configure(image="",text="该组合保存回放待生成；实时3D支持全部控制器和场景。")
        if csv.exists():
            self.plot_image=ImageTk.PhotoImage(curve(csv).resize((760,448)));self.plot.configure(image=self.plot_image,text="")
        else:self.plot.configure(image="",text="保存曲线覆盖fixed/feedback的fast、left、step_stop；其他组合使用原始轨迹。")
        self.metrics.configure(state="normal");self.metrics.delete("1.0","end")
        path=self.folder/"cases.json"
        if path.exists():
            cases=json.loads(path.read_text(encoding="utf-8"));selected=[c for c in cases if c["controller"]==self.mode.get() and c["scenario"]==self.scenario.get()]
            if selected:
                c=selected[0];self.metrics.insert("end",f"冻结结果 · 同场景{sum(r['success'] for r in selected)}/{len(selected)}通过\n种子{c['seed']}：0.5秒均值RMSE v={c['speed_rmse_m_s']:.4f}m/s，yaw={c['yaw_rmse_rad_s']:.4f}rad/s\n原始RMSE v={c['raw_speed_rmse_m_s']:.4f}m/s，yaw={c['raw_yaw_rmse_rad_s']:.4f}rad/s\n失败条件：{[k for k,v in c['criteria'].items() if not v]}\n")
        self.metrics.insert("end","保存GIF/曲线固定首种子；增益输入仅作用于实时与新实验。当前为解析反馈工程基线，RL训练次数0。")
        self.metrics.configure(state="disabled")

    def parameters(self):
        try:
            kv,kw=float(self.kv.get()),float(self.kw.get())
            if not 0<=kv<=5 or not 0<=kw<=5:raise ValueError()
            return ["--velocity-gain",str(kv),"--yaw-gain",str(kw)]
        except ValueError:messagebox.showerror("增益","Kv与Kω需为0到5的有限数值。");return None

    def launch(self,args,newfolder=None):
        if self.process:return
        self.pending=newfolder;logs=ROOT/"outputs/go2-student";logs.mkdir(parents=True,exist_ok=True)
        self.log=logs/(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".log");self.stream=self.log.open("w",encoding="utf-8")
        exe=Path(sys.executable);exe=exe.with_name("python.exe") if exe.name.lower()=="pythonw.exe" else exe
        self.process=subprocess.Popen([str(exe),str(ROOT/"go2_feedback.py"),*args],stdout=self.stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));self.status.configure(text=f"运行中；日志{self.log.name}")

    def live(self):
        if self.process:return
        params=self.parameters()
        if params is not None:self.launch(["--demo","--controller",self.mode.get(),"--scenario",self.scenario.get(),"--seed",self.seed.get(),*params])

    def experiment(self):
        if self.process:return
        params=self.parameters()
        if params is None:return
        folder=ROOT/"outputs/go2-student"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.launch(["--output",str(folder),*params],folder)

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
                if code==0 and self.pending:self.folder=self.pending;self.replay()
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


if __name__=="__main__":Go2Panel().mainloop()

"""Panda residual teaching panel, with fixed replay and explicit live case selection."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk,messagebox
from PIL import Image,ImageTk

ROOT=Path(__file__).resolve().parent
BASELINE=ROOT/"outputs/panda-residual-20261014-r3"


class PandaPanel(tk.Tk):
    def __init__(self):
        super().__init__();self.title("Panda残差放置实验");self.geometry("1120x850");self.minsize(1080,800)
        self.process=self.stream=None;self.frames=[];self.frame=0;self.playing=False
        self.folder=BASELINE;self.protocol("WM_DELETE_WINDOW",self.close)
        style=ttk.Style(self);style.configure("TButton",padding=(10,7),font=("Microsoft YaHei UI",11));style.configure("TLabel",font=("Microsoft YaHei UI",11))
        ttk.Label(self,text="Panda：固定抓取搬运 → XY残差修正 → 释放与停稳验收",font=("Microsoft YaHei UI",17,"bold")).pack(anchor="w",padx=16,pady=12)
        body=ttk.Frame(self);body.pack(fill="both",expand=True,padx=16)
        left=ttk.Frame(body);left.pack(side="left",fill="both",expand=True,padx=(0,16))
        self.picture=ttk.Label(left,anchor="center");self.picture.pack(fill="both",expand=True)
        self.metrics=tk.Text(left,height=11,wrap="word",font=("Microsoft YaHei UI",11),bg="#f8fafc",fg="#0f172a");self.metrics.pack(fill="x",pady=8)
        right=ttk.Frame(body);right.pack(side="right",fill="y")
        ttk.Label(right,text="控制器（实时同条件对照）").pack(anchor="w")
        self.policy=tk.StringVar(value="ppo_seed_0")
        ttk.Combobox(right,textvariable=self.policy,values=["ppo_seed_0","ppo_seed_1","ppo_seed_2","ppo_seed_3","ppo_seed_4","analytical","zero","random","untrained"],state="readonly").pack(fill="x",pady=6)
        ttk.Label(right,text="实时域 / 案例0—7").pack(anchor="w")
        line=ttk.Frame(right);line.pack(fill="x",pady=6)
        self.split,self.case=tk.StringVar(value="test"),tk.StringVar(value="0")
        ttk.Combobox(line,textvariable=self.split,values=["test","ood"],width=10,state="readonly").pack(side="left")
        ttk.Combobox(line,textvariable=self.case,values=list(map(str,range(8))),width=8,state="readonly").pack(side="left",padx=8)
        self.filter=tk.BooleanVar(value=True);ttk.Checkbutton(right,text="启用XY残差变化率过滤",variable=self.filter).pack(anchor="w",pady=6)
        self.buttons=[]
        for label,cmd in (("实时3D · 所选目标放置",self.live),("实时3D · 固定抓取搬运前缀",self.prefix_live),
                          ("载入所选基线回放 · ID案例0",self.replay),("载入抓取搬运前缀回放",self.prefix_replay),
                          ("回放播放 / 暂停",self.toggle),("打开知识与编程操作课",lambda:os.startfile(ROOT/"05-Panda残差放置实操.md")),
                          ("打开原始结果与失败",lambda:os.startfile(self.folder))):
            b=ttk.Button(right,text=label,command=cmd);b.pack(fill="x",pady=4);self.buttons.append(b)
        ttk.Label(right,text="新训练步数 / 单种子").pack(anchor="w",pady=(10,0))
        line=ttk.Frame(right);line.pack(fill="x",pady=6)
        self.steps,self.seed=tk.StringVar(value="8192"),tk.StringVar(value="5")
        ttk.Entry(line,textvariable=self.steps,width=12).pack(side="left");ttk.Entry(line,textvariable=self.seed,width=8).pack(side="left",padx=8)
        self.train_button=ttk.Button(right,text="训练自己的新实验",command=self.train);self.train_button.pack(fill="x",pady=4)
        ttk.Button(right,text="停止当前进程 · 保留结果",command=self.stop).pack(fill="x",pady=4)
        self.status=ttk.Label(self,text="就绪；保存回放为固定ID案例0，实时3D使用右侧域/案例/过滤。",wraplength=1080);self.status.pack(fill="x",padx=16,pady=10)
        self.replay();self.after(160,self.tick)

    def load_gif(self,path):
        self.playing=False;self.frame=0;self.frames=[]
        if path.exists():
            with Image.open(path) as im:
                for i in range(im.n_frames):
                    im.seek(i);self.frames.append(ImageTk.PhotoImage(im.convert("RGB").resize((640,480))))
            self.picture.configure(image=self.frames[0],text="")
        else:self.picture.configure(image="",text="该控制器保存回放待生成；实时3D可运行全部已保存控制器。")

    def replay(self):
        self.load_gif(self.folder/"demos"/(self.policy.get()+".gif"))
        self.metrics.configure(state="normal");self.metrics.delete("1.0","end")
        path=self.folder/"summary.json"
        if path.exists():
            record=json.loads(path.read_text(encoding="utf-8"))["aggregate"].get(self.policy.get())
            if record:self.metrics.insert("end",f"固定基线 · ID8案例 · 开启过滤\n成功{record['successes']}/{record['cases']}；平均放置误差{record['mean_error_m']*1000:.2f}mm；违规{record['violations']}\n")
        self.metrics.insert("end","保存GIF为ID案例0/过滤开启。实时选择独立生效。训练从9秒抓取搬运快照开始，夹爪与高度轨迹由名义控制器给定。\n策略使用仿真状态输入；任务为目标平移适配。")
        self.metrics.configure(state="disabled")

    def prefix_replay(self):
        self.load_gif(ROOT/"outputs/panda-prefix-20261014/demo.gif")
        self.status.configure(text="固定前缀0—9秒：接近、下降、夹紧、抬升、搬运；放置残差课从其末态开始。")

    def launch(self,script,args,training=False):
        if self.process:return
        logs=ROOT/"outputs/panda-student";logs.mkdir(parents=True,exist_ok=True)
        self.log=logs/(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")+".log");self.stream=self.log.open("w",encoding="utf-8")
        exe=Path(sys.executable);exe=exe.with_name("python.exe") if exe.name.lower()=="pythonw.exe" else exe
        self.process=subprocess.Popen([str(exe),str(ROOT/script),*args],stdout=self.stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        self.training=training;self.train_button.configure(state="disabled");self.status.configure(text=f"运行中；日志{self.log.name}")

    def live(self):
        args=["--demo",str(self.folder),"--policy",self.policy.get(),"--split",self.split.get(),"--case",self.case.get()]
        if not self.filter.get():args.append("--filter-off")
        self.launch("panda_residual.py",args)

    def prefix_live(self):self.launch("panda_prefix_demo.py",["--live"])

    def train(self):
        if self.process:return
        try:
            steps,seed=int(self.steps.get()),int(self.seed.get())
            if not 0<steps<=131072 or steps%1024 or seed<0:raise ValueError()
        except ValueError:
            messagebox.showerror("参数","步数为1024正整数倍，最大131072；种子为非负整数。");return
        self.pending=ROOT/"outputs/panda-student"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f");self.pending_seed=seed
        self.launch("panda_residual.py",["--output",str(self.pending),"--steps",str(steps),"--seeds",str(seed)],True)

    def toggle(self):self.playing=not self.playing

    def tick(self):
        if self.playing and self.frames:
            self.frame=(self.frame+1)%len(self.frames);self.picture.configure(image=self.frames[self.frame])
        if self.process:
            code=self.process.poll()
            if code is None:
                lines=self.log.read_text(encoding="utf-8",errors="replace").splitlines()
                if lines:self.status.configure(text=f"运行中：{lines[-1]}")
            else:
                self.stream.close();self.process=self.stream=None;self.train_button.configure(state="normal")
                if code==0 and self.training:
                    self.folder=self.pending;self.policy.set(f"ppo_seed_{self.pending_seed}");self.replay()
                self.status.configure(text=f"完成，退出码{code}；日志{self.log.name}")
        self.after(160,self.tick)

    def stop(self):
        if self.process:self.process.terminate()
        self.playing=False

    def close(self):
        self.stop()
        if self.process:self.process.wait(timeout=5)
        if self.stream:self.stream.close()
        self.destroy()


if __name__=="__main__":PandaPanel().mainloop()

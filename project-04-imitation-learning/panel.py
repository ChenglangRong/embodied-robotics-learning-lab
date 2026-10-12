"""Native step-by-step learning panel: data, training, closed-loop control and evidence."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import webbrowser
import numpy as np
from learning import ROOT, MLP, TwoLinkReachEnv, expert, encode, save_json, write_csv
from visuals import arm_points


class LearningPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("模仿学习实验室 · 项目04")
        self.geometry("1080x820")
        self.minsize(940,740)
        self.running=False
        self.process=None
        self.model=None
        self.trace=[]
        self.ghost_done=False
        self.current_spec=None
        self.closed=False
        self.protocol("WM_DELETE_WINDOW",self.close)
        style=ttk.Style(self)
        style.configure("TButton",padding=(10,6))
        style.configure("TLabel",font=("Microsoft YaHei UI",11))
        title=ttk.Label(self,text="专家示范 → 行为克隆 → 策略独立执行",font=("Microsoft YaHei UI",18,"bold"))
        title.pack(anchor="w",padx=20,pady=(18,6))
        ttk.Label(self,text="观测6维 · 关节速度动作2维 · 50 Hz运动学环境 · 同一1 mm到达判据").pack(anchor="w",padx=20)
        self.run_label=ttk.Label(self,text="载入结果…",font=("Microsoft YaHei UI",10))
        self.run_label.pack(anchor="w",padx=20,pady=(4,0))
        # Reserve the footer before allocating expanding body space.
        self.status=ttk.Label(self,text="就绪",wraplength=1000)
        self.status.pack(side="bottom",anchor="w",padx=20,pady=(0,12))
        self.progress=ttk.Progressbar(self,mode="indeterminate")
        self.progress.pack(side="bottom",fill="x",padx=20,pady=(0,6))
        body=ttk.Frame(self)
        body.pack(fill="both",expand=True,padx=20,pady=12)
        body.columnconfigure(0,weight=1)
        body.columnconfigure(1,weight=0)
        body.rowconfigure(0,weight=1)
        left=ttk.Frame(body)
        left.grid(row=0,column=0,sticky="nsew",padx=(0,18))
        self.canvas=tk.Canvas(left,width=640,height=430,bg="#f8fbfc",highlightthickness=1,highlightbackground="#a9c0c9")
        self.canvas.pack(fill="both",expand=True)
        self.canvas.bind("<Configure>",lambda _:self.draw())
        self.readout=tk.Text(left,height=7,font=("Consolas",11),wrap="word",bg="#edf4f5",fg="#172b3a")
        self.readout.pack(fill="x",pady=(10,0))
        right_shell=ttk.Frame(body,width=310)
        right_shell.grid(row=0,column=1,sticky="ns")
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
        def reveal(event):
            widget=event.widget
            ancestor=widget
            while ancestor is not None and ancestor is not right:
                ancestor=ancestor.master
            if ancestor is None:return
            top=widget.winfo_rooty()-right.winfo_rooty()
            visible=controls.canvasy(0)
            if top<visible or top+widget.winfo_height()>visible+controls.winfo_height():
                controls.yview_moveto(max(0,top-16)/max(right.winfo_height(),1))
        right.bind_all("<FocusIn>",reveal,add="+")
        self.worker_buttons=[]
        for text,mode,collect_only in (("1  收集专家示范", "quick",True),("2  快速训练（1种子）","quick",False),("3  完整比较（5种子）","full",False)):
            button=ttk.Button(right,text=text,command=lambda m=mode,c=collect_only:self.start_worker(m,c))
            button.pack(fill="x",pady=(0,8))
            self.worker_buttons.append(button)
        ttk.Label(right,text="4  选择闭环控制器").pack(anchor="w",pady=(8,4))
        self.policy_name=tk.StringVar(value="expert")
        self.policy_box=ttk.Combobox(right,textvariable=self.policy_name,state="readonly",values=["expert","zero","bc","bc_more_steps","bc_more_data","dagger"])
        self.policy_box.pack(fill="x")
        self.policy_box.bind("<<ComboboxSelected>>",lambda _:self.reset())
        ttk.Label(right,text="评测案例：ID / OOD").pack(anchor="w",pady=(8,4))
        self.case_name=tk.StringVar()
        self.case_box=ttk.Combobox(right,textvariable=self.case_name,state="readonly")
        self.case_box.pack(fill="x")
        self.case_box.bind("<<ComboboxSelected>>",lambda _:self.choose_case())
        buttons=ttk.Frame(right)
        buttons.pack(fill="x",pady=10)
        ttk.Button(buttons,text="播放 / 暂停",command=self.toggle).pack(side="left",fill="x",expand=True)
        ttk.Button(buttons,text="单步",command=lambda:self.step(manual=True)).pack(side="left",fill="x",expand=True,padx=(6,0))
        ttk.Button(right,text="重置同一起点与目标",command=self.reset).pack(fill="x",pady=(0,8))
        ttk.Button(right,text="载入已测五种子基线",command=self.load_baseline).pack(fill="x",pady=(0,6))
        custom=ttk.LabelFrame(right,text="自己的目标（m）",padding=8)
        custom.pack(fill="x",pady=6)
        self.goal_x,self.goal_y=tk.StringVar(value="0.55"),tk.StringVar(value="0.55")
        ttk.Label(custom,text="x").grid(row=0,column=0)
        ttk.Entry(custom,textvariable=self.goal_x,width=8).grid(row=0,column=1,padx=5)
        ttk.Label(custom,text="y").grid(row=0,column=2)
        ttk.Entry(custom,textvariable=self.goal_y,width=8).grid(row=0,column=3,padx=5)
        ttk.Button(custom,text="使用这个目标（seed=11）",command=self.custom_case).grid(row=1,column=0,columnspan=4,sticky="ew",pady=(8,0))
        ttk.Button(right,text="保存自己的实验记录",command=self.save_personal).pack(fill="x",pady=6)
        ttk.Button(right,text="打开结果图与报告",command=self.open_report).pack(fill="x",pady=6)
        ttk.Button(right,text="打开逐步实操课",command=lambda:os.startfile(ROOT/"02-行为克隆实操课.md")).pack(fill="x",pady=6)
        latest=ROOT/"outputs/latest.txt"
        candidate=Path(latest.read_text(encoding="utf-8").strip()) if latest.exists() else None
        self.run_dir=candidate if candidate and (candidate/"summary.json").exists() else ROOT/"outputs/baseline-20261012-r2"
        self.specs=[]
        self.load_run()
        self.after(20,self.tick)

    def set_status(self,text):
        self.status.configure(text=text)

    def load_run(self):
        summary_path=self.run_dir/"summary.json"
        if summary_path.exists():
            summary=json.loads(summary_path.read_text(encoding="utf-8"))
            config=summary.get("training_config",{})
            self.run_label.configure(text=f"当前权重：{summary['mode']} / {len(config.get('seeds',[]))}种子 / 演示seed=0；基础{config.get('base_updates',0)}、追加{config.get('branch_updates',0)}次更新")
        protocol_path=self.run_dir/"protocol.json"
        if protocol_path.exists():
            data=json.loads(protocol_path.read_text(encoding="utf-8"))
            self.specs=data["test"]+data["ood"]
        else:
            from learning import protocol
            data=protocol()
            self.specs=data["test"]+data["ood"]
        self.case_box.configure(values=[s["id"] for s in self.specs])
        self.case_name.set(self.specs[0]["id"])
        self.current_spec=self.specs[0]
        self.reset()
        self.set_status(f"当前结果：{self.run_dir}；先播放expert，再选择bc和dagger进行同条件比较。")

    def load_baseline(self):
        if self.process and self.process.poll() is None:
            self.set_status("当前训练完成后再载入正式基线。")
            return
        self.run_dir=ROOT/"outputs/baseline-20261012-r2"
        self.load_run()

    def choose_case(self):
        self.current_spec=next(s for s in self.specs if s["id"]==self.case_name.get())
        self.reset()

    def reset(self):
        self.running=False
        if self.current_spec is None:return
        self.env=TwoLinkReachEnv()
        self.env.reset(seed=self.current_spec["reset_seed"],goal=self.current_spec["goal"])
        self.ghost=TwoLinkReachEnv()
        self.ghost.reset(seed=self.current_spec["reset_seed"],goal=self.current_spec["goal"])
        self.ghost_done=False
        self.trace=[]
        self.done=False
        self.model=None
        name=self.policy_name.get()
        if name not in ("expert","zero"):
            checkpoint=self.run_dir/"seed-0"/f"{name}.npz"
            if checkpoint.exists():self.model=MLP.load(checkpoint)
            else:
                self.set_status("该策略权重尚待训练：点击快速训练，完成后再选择学习策略。")
        self.draw()
        self.show_state()

    def custom_case(self):
        try:
            goal=np.array([float(self.goal_x.get()),float(self.goal_y.get())])
            if not np.all(np.isfinite(goal)):raise ValueError("坐标需要有限数值")
        except ValueError as error:
            self.set_status(f"目标输入：{error}")
            return
        self.current_spec={"id":"personal-goal","split":"personal","reset_seed":11,"goal":goal.tolist()}
        self.case_name.set("personal-goal")
        self.reset()

    def toggle(self):
        if self.done:
            self.reset()
        if self.policy_name.get() not in ("expert","zero") and self.model is None:
            self.set_status("完成训练后即可播放这个学习策略。")
            return
        self.running=not self.running

    def step(self,manual=False):
        if manual:self.running=False
        if self.done:return
        name=self.policy_name.get()
        if name=="expert":action=expert(self.env)
        elif name=="zero":action=np.zeros(2)
        elif self.model is not None:action=self.model(self.env)
        else:return
        _,_,done,info=self.env.step(action)
        if not self.ghost_done:
            _,_,self.ghost_done,_=self.ghost.step(expert(self.ghost))
        self.trace.append({"step":self.env.step_count,"time_s":self.env.step_count*self.env.cfg.dt,"q1":self.env.q[0],"q2":self.env.q[1],
                           "x_m":self.env.position()[0],"y_m":self.env.position()[1],"error_m":info["error"],
                           "qdot1_rad_s":info["applied_action"][0],"qdot2_rad_s":info["applied_action"][1],"action_clipped":info["action_clipped"]})
        self.done=done
        if done:
            self.running=False
            self.set_status(f"{name}：{'到达成功' if info['success'] else '达到600步上限'}；最终误差{info['error']*1000:.3f} mm。")
        self.draw()
        self.show_state()

    def tick(self):
        if self.closed:return
        if self.running:self.step()
        self.after(20,self.tick)

    def draw(self):
        if not hasattr(self,"env"):return
        canvas=self.canvas
        canvas.delete("all")
        width,height=max(canvas.winfo_width(),400),max(canvas.winfo_height(),300)
        origin=np.array([width/2,height/2])
        scale=min(width,height)*.43
        def pixel(point):return tuple(origin+np.asarray(point)*[scale,-scale])
        for radius in (.2,1.):
            r=radius*scale
            canvas.create_oval(origin[0]-r,origin[1]-r,origin[0]+r,origin[1]+r,outline="#c6d8df",dash=(3,4))
        for points,color,line_width,dash in ((arm_points(self.ghost.q),"#8299a4",3,(6,4)),(arm_points(self.env.q),"#0f766e",6,None)):
            coords=[v for p in points for v in pixel(p)]
            canvas.create_line(*coords,fill=color,width=line_width,dash=dash)
        trail=[pixel([r["x_m"],r["y_m"]]) for r in self.trace]
        if len(trail)>1:canvas.create_line(*[v for p in trail for v in p],fill="#0f766e",width=1)
        gx,gy=pixel(self.env.goal)
        canvas.create_line(gx-8,gy,gx+8,gy,fill="#b91c1c",width=3)
        canvas.create_line(gx,gy-8,gx,gy+8,fill="#b91c1c",width=3)
        canvas.create_text(gx+12,gy,text="目标",anchor="w",fill="#b91c1c",font=("Microsoft YaHei UI",11))
        canvas.create_text(16,18,text=f"实线：{self.policy_name.get()}     虚线：专家参照",anchor="w",fill="#172b3a",font=("Microsoft YaHei UI",11))

    def show_state(self):
        observation=encode(self.env)
        error=np.linalg.norm(self.env.goal-self.env.position())
        text=f"模拟时间 {self.env.step_count*self.env.cfg.dt:.2f} s | 位置误差 {error*1000:.3f} mm | 目标阈值 {self.env.cfg.tolerance*1000:g} mm\n"
        text+=f"obs[6] = {np.array2string(observation,precision=3)}\n"
        text+=f"q = {np.array2string(self.env.q,precision=3)} rad\n"
        if self.trace:text+=f"action[2] = [{self.trace[-1]['qdot1_rad_s']:.3f}, {self.trace[-1]['qdot2_rad_s']:.3f}] rad/s\n"
        text+="每次单步：读取obs → 输出速度 → 限幅 → 更新关节 → 再次观测。"
        self.readout.configure(state="normal")
        self.readout.delete("1.0","end")
        self.readout.insert("1.0",text)
        self.readout.configure(state="disabled")

    def start_worker(self,mode,collect_only):
        if self.process and self.process.poll() is None:return
        self.running=False
        stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.worker_dir=ROOT/"outputs/runs"/stamp
        log_dir=ROOT/"outputs/console"
        log_dir.mkdir(parents=True,exist_ok=True)
        self.log_path=log_dir/f"{stamp}.log"
        self.log=self.log_path.open("w",encoding="utf-8")
        py=Path(sys.executable)
        if py.name.lower()=="pythonw.exe":py=py.with_name("python.exe")
        args=[str(py),str(ROOT/"learning.py"),"--mode",mode,"--output",str(self.worker_dir)]
        if collect_only:args.append("--collect-only")
        try:
            self.process=subprocess.Popen(args,cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        except OSError as error:
            self.log.close()
            self.set_status(f"启动失败：{error}；核对Python路径后重试。")
            return
        for button in self.worker_buttons:button.configure(state="disabled")
        self.progress.start(15)
        self.set_status("正在收集与训练；下方状态将显示当前阶段、更新步数和评测结果。")
        self.after(500,self.poll_worker)

    def poll_worker(self):
        if self.closed:return
        lines=self.log_path.read_text(encoding="utf-8",errors="replace").splitlines()
        if lines:self.set_status(lines[-1])
        if self.process.poll() is None:
            self.after(500,self.poll_worker)
            return
        self.log.close()
        self.progress.stop()
        for button in self.worker_buttons:button.configure(state="normal")
        if self.process.returncode:
            self.set_status(f"运行出错；日志保留于{self.log_path}")
            self.readout.configure(state="normal")
            self.readout.delete("1.0","end")
            self.readout.insert("1.0","\n".join(lines[-15:]))
            self.readout.configure(state="disabled")
            return
        self.run_dir=self.worker_dir
        self.load_run()
        if (self.run_dir/"report.html").exists():self.open_report()
        else:
            summary=json.loads((self.run_dir/"summary.json").read_text(encoding="utf-8"))
            self.set_status(f"专家数据已收集：{summary['data_counts']}；训练按钮将创建新的完整实验目录。")

    def open_report(self):
        report=self.run_dir/"report.html"
        if report.exists():webbrowser.open(report.resolve().as_uri())
        else:self.set_status("训练完成后生成图与报告；当前可先播放expert示范。")

    def save_personal(self):
        if not self.trace:
            self.set_status("先播放或单步推进，再保存自己的运行。")
            return
        path=ROOT/"outputs/student"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path.mkdir(parents=True)
        write_csv(path/"trajectory.csv",self.trace)
        checkpoint=self.run_dir/"seed-0"/f"{self.policy_name.get()}.npz"
        save_json(path/"config.json",{"spec":self.current_spec,"policy":self.policy_name.get(),"run_directory":str(self.run_dir),
                                     "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.exists() else None,
                                     "completed":self.done,"last_error_m":self.trace[-1]["error_m"]})
        self.set_status(f"个人实验已保存：{path}")

    def close(self):
        if self.process and self.process.poll() is None:
            self.set_status("训练仍在运行；完成后再关闭面板，保留完整日志与结果。")
            return
        self.closed=True
        self.destroy()


if __name__=="__main__":
    LearningPanel().mainloop()

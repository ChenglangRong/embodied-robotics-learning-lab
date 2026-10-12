"""Scientific figures and genuine environment rollouts, generated from saved runs."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from learning import MLP, TwoLinkReachEnv, expert, rollout, save_json

COLORS = ("#2563eb", "#c2410c", "#15803d", "#7c3aed")
METHODS = ("bc", "bc_more_steps", "bc_more_data", "dagger")
INK, MUTED, GRID = "#172b3a", "#526575", "#d6e0e5"


def font(size):
    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def chart(draw, box, title, series, xlabel, ylabel, bars=False, limits=None):
    x0, y0, width, height = box
    left, right, top, bottom = x0+75, x0+width-25, y0+65, y0+height-80
    draw.text((x0+12, y0+5), title, font=font(22), fill=INK)
    draw.text((x0+12, y0+35), ylabel, font=font(16), fill=MUTED)
    values = [y for _, pts, _ in series for _, y in pts]
    xmax = max(x for _, pts, _ in series for x, _ in pts)
    if bars:
        xmax += .7
    lo, hi = limits if limits else (min(0, min(values)), max(values)*1.08+.00001)
    def xy(x,y):
        return left+x/max(xmax, 1e-9)*(right-left), bottom-(y-lo)/(hi-lo)*(bottom-top)
    for i in range(5):
        y = lo+(hi-lo)*i/4
        _, py = xy(0,y)
        draw.line((left,py,right,py), fill=GRID)
        draw.text((left-8,py), f"{y:.3g}", font=font(16), fill=MUTED, anchor="rm")
    draw.rectangle((left,top,right,bottom), outline=MUTED)
    for name, points, color in series:
        if bars:
            for x,y in points:
                px,py = xy(x,y)
                draw.rectangle((px-14, min(py,bottom), px+14, bottom), fill=color)
                draw.text((px,py-6), f"{y:.3g}", font=font(14), fill=INK, anchor="mb")
        else:
            draw.line([xy(x,y) for x,y in points], fill=color, width=2)
    if bars:
        for i,name in enumerate(METHODS,1):
            px,_=xy(i,0)
            label = {"bc":"BC", "bc_more_steps":"+updates", "bc_more_data":"+expert", "dagger":"DAgger"}[name]
            draw.text((px,bottom+12), label, font=font(15), fill=INK, anchor="mt")
    else:
        for i in range(5):
            x=xmax*i/4
            px,_=xy(x,lo)
            draw.text((px,bottom+12), f"{x:.0f}", font=font(15), fill=INK, anchor="mt")
    draw.text((left,bottom+40), xlabel, font=font(15), fill=MUTED)
    for i,(name,_,color) in enumerate(series):
        draw.line((x0+width-210,y0+8+i*21,x0+width-190,y0+8+i*21),fill=color,width=3)
        draw.text((x0+width-182,y0-2+i*21),name,font=font(15),fill=INK)


def create_figure(root, summary):
    image=Image.new("RGB",(1440,970),"white")
    draw=ImageDraw.Draw(image)
    count=len(summary["results"])
    draw.text((35,20),"Project 04 | Behavior cloning and closed-loop evaluation",font=font(30),fill=INK)
    draw.text((35,65),f"{count} training seed(s), fixed 24 ID + 24 OOD episodes; success tolerance = 1 mm",font=font(19),fill=MUTED)
    seed=next(iter(summary["results"]))
    with (root/f"seed-{seed}/bc-loss.csv").open(encoding="utf-8") as stream:
        history=list(csv.DictReader(stream))
    curves=[("train",[(float(r["update"]),float(r["train_mse"])) for r in history],COLORS[0]),
            ("validation",[(float(r["update"]),float(r["validation_mse"])) for r in history],COLORS[1])]
    chart(draw,(25,110,690,400),f"BC loss (seed {seed})",curves,"optimizer update","MSE (rad/s)^2")
    rates=[]
    for split,color,offset in (("test",COLORS[0],-.18),("ood",COLORS[1],.18)):
        rates.append(("ID" if split=="test" else "OOD",[(i+offset,summary["aggregate"][m][split]["mean_success_rate"]) for i,m in enumerate(METHODS,1)],color))
    chart(draw,(730,110,690,400),"Closed-loop success",rates,"same 1 mm criterion","success fraction",bars=True,limits=(0,1.12))
    teacher=[("ID",[(i,float(np.mean([r[m]["teacher_forced_test_mse"] for r in summary["results"].values()]))) for i,m in enumerate(METHODS,1)],COLORS[0])]
    chart(draw,(25,550,690,400),"Teacher-forced action error",teacher,"expert-visited states","MSE (rad/s)^2",bars=True)
    errors=[("ID",[(i,summary["aggregate"][m]["test"]["mean_final_error_m"]*1000) for i,m in enumerate(METHODS,1)],COLORS[0])]
    chart(draw,(730,550,690,400),"Closed-loop final error",errors,"learner-visited states","mean position error (mm)",bars=True)
    image.save(root/"overview.png")


def arm_points(q, lengths=(.6,.4)):
    elbow=np.array([lengths[0]*np.cos(q[0]), lengths[0]*np.sin(q[0])])
    end=elbow+np.array([lengths[1]*np.cos(q.sum()),lengths[1]*np.sin(q.sum())])
    return np.array([[0.,0.],elbow,end])


def create_demo(root, summary, split="test"):
    seed=next(iter(summary["results"]))
    spec=summary["protocol"][split][0]  # Frozen first case; no best-case selection.
    policies=[("Expert",expert,COLORS[2]),("BC",MLP.load(root/f"seed-{seed}/bc.npz"),COLORS[0]),
              ("DAgger",MLP.load(root/f"seed-{seed}/dagger.npz"),COLORS[1])]
    rollouts=[rollout(spec,p) for _,p,_ in policies]
    env=TwoLinkReachEnv()
    env.reset(seed=spec["reset_seed"],goal=spec["goal"])
    initial=env.q.copy()
    frames=[]
    horizon=max(len(trace) for _,trace in rollouts)
    for step in range(0,horizon,4):
        frame=Image.new("RGB",(1080,420),"white")
        draw=ImageDraw.Draw(frame)
        draw.text((20,12),f"{split.upper()} | {spec['id']} | same start/goal | t={step*.02:.2f}s",font=font(22),fill=INK)
        for index,((name,_,color),(result,trace)) in enumerate(zip(policies,rollouts)):
            x0=index*360
            draw.text((x0+15,50),name,font=font(23),fill=color)
            draw.text((x0+15,83),f"{result['termination']} | final {result['final_error_m']*1000:.2f} mm",font=font(15),fill=INK)
            origin=np.array([x0+180.,260.])
            def scale(point):
                return tuple(origin+np.asarray(point)*np.array([135.,-135.]))
            draw.ellipse((origin[0]-135,origin[1]-135,origin[0]+135,origin[1]+135),outline=GRID)
            record=trace[min(step,len(trace)-1)]
            q=np.array([record["q1"],record["q2"]])
            points=[scale(p) for p in arm_points(q)]
            draw.line(points,fill=color,width=7)
            for px,py in points:
                draw.ellipse((px-5,py-5,px+5,py+5),fill=color)
            gx,gy=scale(spec["goal"])
            draw.line((gx-7,gy,gx+7,gy),fill="#b91c1c",width=2)
            draw.line((gx,gy-7,gx,gy+7),fill="#b91c1c",width=2)
            trail=[scale([r["x_m"],r["y_m"]]) for r in trace[:min(step+1,len(trace))]]
            if len(trail)>1:
                draw.line(trail,fill=color,width=1)
        frames.append(frame.quantize(colors=64))
    frames[0].save(root/f"demo-{split}.gif",save_all=True,append_images=frames[1:],duration=80,loop=0)
    save_json(root/f"demo-{split}-selection.json",{"spec":spec,"seed":seed,"selection":"first frozen evaluation case",
              "results":{name:result for (name,_,_),(result,_) in zip(policies,rollouts)}})


def create_report(root, summary):
    count=len(summary["results"])
    text=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>项目04训练结果</title><style>body{font:17px system-ui;line-height:1.65;max-width:1100px;margin:24px auto;padding:0 18px;color:#172b3a;background:#fff}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #b4c8ce;padding:10px;text-align:left}a{color:#155e75}pre{white-space:pre-wrap;background:#f0f6f6;padding:16px}.table-wrap{overflow:auto}</style><h1>项目04：行为克隆与闭环执行</h1>']
    text.append(f'<p>{count}个训练种子；每种子固定24个ID与24个OOD目标；成功容差1 mm，最长600步。ID目标角度0.10–1.30 rad，OOD为−1.30–−0.10 rad。数字是本次实验测量，原始结果见<a href="summary.json">summary.json</a>。</p>')
    text.append('<p>BC：基础训练；+updates：同数据追加训练；+expert：追加新专家轨迹；DAgger：追加学习策略访问状态的专家标签。三个追加分支具有相同更新步数，后两者新增标签数一致；环境步数与壁钟成本分别记录。五种子固定测试集衡量训练随机性，跨数据集和实机泛化另行验收。</p>')
    text.append('<div class="table-wrap"><table><tr><th>方法</th><th>ID成功率均值±训练种子SD</th><th>OOD成功率均值±SD</th><th>ID最终误差均值(mm)</th></tr>')
    for name in ("expert","zero",*METHODS):
        if name in summary["baselines"]:
            source=summary["baselines"][name]
            cells=[f'{source[s]["success_rate"]:.1%}' for s in ("test","ood")]
            error=source["test"]["mean_final_error_m"]*1000
        else:
            source=summary["aggregate"][name]
            cells=[]
            for s in ("test","ood"):
                sd=source[s]["sample_std_success_rate"]
                cells.append(f'{source[s]["mean_success_rate"]:.1%}'+(f' ± {sd:.1%}' if sd is not None else '（单种子）'))
            error=source["test"]["mean_final_error_m"]*1000
        text.append(f'<tr><td>{name}</td><td>{cells[0]}</td><td>{cells[1]}</td><td>{error:.3f}</td></tr>')
    text.append('</table></div><h2>训练与验收图</h2><img src="overview.png" alt="训练损失、单步动作误差、闭环成功率与最终误差"><h2>同起点同目标的真实闭环回放</h2><p>演示使用冻结测试集第一项；每条轨迹由对应控制器独立执行。先看专家，再看BC和DAgger，比较运动方向、停留位置和最终误差。</p><img src="demo-test.gif" alt="专家、BC与DAgger的ID闭环运动"><h2>目标域外回放</h2><img src="demo-ood.gif" alt="专家、BC与DAgger的OOD闭环运动"><h2>本节训练题</h2><ol><li>找一个单步误差较小而闭环失败的案例，读取最终位置误差。</li><li>核对query-ledger.json中的DAgger与新专家轨迹标签数。</li><li>复制配置，只改训练步数或示范数量，重新运行并记录差异。</li><li>解释为何目标分布改变后，测试结果需要单独报告。</li></ol><p>这是二连杆运动学算法实验；相机、动力学接触与Panda操作在后续闸门中增加。</p></html>')
    (root/"report.html").write_text("".join(text),encoding="utf-8")


def create_outputs(output):
    root=Path(output)
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    create_figure(root,summary)
    create_demo(root,summary,"test")
    create_demo(root,summary,"ood")
    create_report(root,summary)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    create_outputs(parser.parse_args().output)

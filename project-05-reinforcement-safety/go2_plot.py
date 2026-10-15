"""Render raw, causal-window and command traces from a saved episode CSV."""
import argparse
import csv
from pathlib import Path
from PIL import Image,ImageDraw


def curve(path):
    with path.open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
    im=Image.new("RGB",(900,530),"#f8fafc");d=ImageDraw.Draw(im)
    d.text((35,12),path.stem+" | command / raw measurement / trailing 0.5s average",fill="#0f172a")
    for panel,(label,keys) in enumerate((("Body forward velocity (m/s)",("command_v_m_s","v_body_m_s","cycle_v_m_s")),("World yaw rate (rad/s)",("command_yaw_rad_s","yaw_rate_rad_s","cycle_yaw_rad_s")))):
        x0,x1,y0,y1=65,865,55+panel*235,240+panel*235
        values=[[float(r[k]) for r in rows] for k in keys];lo=min(min(v) for v in values);hi=max(max(v) for v in values);pad=max(.025,(hi-lo)*.1);lo-=pad;hi+=pad
        d.text((x0,y0-20),label,fill="#0f172a")
        for j in range(5):
            y=y0+(y1-y0)*j/4;v=hi-(hi-lo)*j/4
            d.line((x0,y,x1,y),fill="#cbd5e1");d.text((5,y-5),f"{v:.2f}",fill="#334155")
        for v,color,width in zip(values,("#c2410c","#64748b","#1d4ed8"),(3,1,3)):
            points=[(x0+float(r["time_s"])/10*(x1-x0),y1-(a-lo)/(hi-lo)*(y1-y0)) for r,a in zip(rows,v)]
            if len(points)>1:d.line(points,fill=color,width=width)
        d.text((x0,y1+3),"0",fill="#334155");d.text((x1-55,y1+3),"10s",fill="#334155")
    for i,(text,col) in enumerate((("command","#c2410c"),("raw","#64748b"),("0.5s average","#1d4ed8"))):
        d.line((150+220*i,514,175+220*i,514),fill=col,width=3);d.text((181+220*i,508),text,fill="#0f172a")
    return im


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("csv",type=Path);p.add_argument("output",type=Path);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);curve(a.csv).save(a.output)

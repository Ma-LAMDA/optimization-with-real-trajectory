from pathlib import Path
import subprocess,time,json,os,re,datetime
base=Path("/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/control")
out=base/"gpu_idle_120s.json"
samples=[]
ok_all=True
for i in range(13):
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    g=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True,capture_output=True)
    c=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True,capture_output=True)
    p=subprocess.run(["bash","-lc","ps -eo pid,cmd | grep -E 'vllm serve|run_codex_ip_trajectories|codex exec|torchrun|deepspeed|swift sft|llamafactory|train.py' | grep -v grep"],text=True,capture_output=True)
    rows=[]
    gpu_ok=(g.returncode==0)
    for line in g.stdout.strip().splitlines():
        try:
            idx,mem,util=[int(x.strip()) for x in line.split(",")]
            rows.append({"index":idx,"memory_used_mib":mem,"utilization_pct":util})
            gpu_ok=gpu_ok and mem<=1024 and util<=1
        except Exception:
            gpu_ok=False
    gpu_ok=gpu_ok and len(rows)==2
    compute=[x for x in c.stdout.strip().splitlines() if x.strip()]
    procs=[x for x in p.stdout.strip().splitlines() if x.strip()]
    sample_ok=gpu_ok and not compute and not procs
    ok_all=ok_all and sample_ok
    samples.append({"at":now,"ok":sample_ok,"gpus":rows,"compute":compute,"relevant_processes":procs})
    tmp=out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"complete":False,"idle_so_far":ok_all,"samples":samples},ensure_ascii=False,indent=2)+"\n")
    os.replace(tmp,out)
    if i<12: time.sleep(10)
final={"complete":True,"continuous_idle_120s":ok_all,"started_at":samples[0]["at"],"finished_at":samples[-1]["at"],"samples":samples}
tmp=out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
os.replace(tmp,out)

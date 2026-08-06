#!/usr/bin/env python3
import json,sys,statistics
from pathlib import Path
run=Path(sys.argv[1])
losses={1:0.23613213002681732,2:0.16515934467315674,3:0.15305212140083313,4:0.14917336404323578,5:0.1493201106786728}
expected_cases=[12,20,38,71,86,100]
rows=[]
for epoch,step in enumerate((40,80,120,160,200),1):
    p=run/"checkpoint_selection"/f"epoch-{epoch}"/"report"/"validation_summary.json"
    d=json.load(open(p))
    overall=d["overall"]
    if d.get("status")!="completed" or sorted(d["case_ids"])!=expected_cases:
        raise SystemExit(f"invalid summary for epoch {epoch}: status/cases")
    if overall["attempts"]!=12 or d["repeats_per_case"]!=2:
        raise SystemExit(f"invalid summary for epoch {epoch}: attempts/repeats")
    if overall.get("attempts_with_captured_reasoning")!=12 or overall.get("reasoning_items",0)<=0:
        raise SystemExit(f"invalid summary for epoch {epoch}: reasoning")
    item={
      "epoch":epoch,"step":step,"checkpoint":str(run/"train"/f"checkpoint-{step}"),
      "model":f"Qwen3.6-27B-0804-e{epoch}",
      "strict_correct":overall["strict_correct"],"attempts":overall["attempts"],
      "accuracy_percent":overall["accuracy_percent"],"model_hard_timeouts":overall["timeouts"],
      "mean_runtime_minutes":overall["runtime_minutes"]["mean"],
      "eval_loss":losses[epoch],"summary":str(p),
      "reasoning_items":overall["reasoning_items"],
      "attempts_with_captured_reasoning":overall["attempts_with_captured_reasoning"],
    }
    item["selection_key"]=[-item["strict_correct"],item["model_hard_timeouts"],item["mean_runtime_minutes"],item["eval_loss"],item["epoch"]]
    rows.append(item)
rows.sort(key=lambda x:tuple(x["selection_key"]))
out={"schema_version":"0804-checkpoint-selection.v1","rule":["strict_accuracy_desc","model_hard_timeouts_asc","mean_runtime_asc","eval_loss_asc","epoch_asc"],"candidates":rows,"selected":rows[0]}
p=run/"checkpoint_selection"/"selection_summary.json"
p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
(run/"checkpoint_selection"/"selected_epoch.txt").write_text(str(rows[0]["epoch"])+"\n")
(run/"checkpoint_selection"/"selected_model.txt").write_text(rows[0]["model"]+"\n")
(run/"checkpoint_selection"/"selected_checkpoint.txt").write_text(rows[0]["checkpoint"]+"\n")
print(json.dumps(rows[0],ensure_ascii=False))

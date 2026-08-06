#!/usr/bin/env python3
import json,sys,statistics,math,csv
from pathlib import Path
run=Path(sys.argv[1])
sel=json.load(open(run/"checkpoint_selection"/"selection_summary.json"))["selected"]
chosen=json.load(open(sel["summary"]))
extra=json.load(open(run/"final_validation"/"selected_extra"/"report"/"validation_summary.json"))
other=json.load(open(run/"final_validation"/"nonselection"/"report"/"validation_summary.json"))
rows=[]
for source,offset,used in [(chosen,0,True),(extra,2,False),(other,0,False)]:
    for item in source["runs"]:
        row=dict(item)
        row["repeat"]=int(row["repeat"])+offset
        row["used_for_checkpoint_selection"]=used
        rows.append(row)
rows.sort(key=lambda x:(int(x["case_id"]),int(x["repeat"])))
keys=[(int(x["case_id"]),int(x["repeat"])) for x in rows]
if len(rows)!=60 or len(keys)!=len(set(keys)):
    raise SystemExit(f"expected 60 unique runs, got {len(rows)} / {len(set(keys))}")
expected_cases=[2,12,19,20,29,38,65,71,85,86,99,100]
if sorted({int(x["case_id"]) for x in rows})!=expected_cases:
    raise SystemExit("final case set mismatch")
if any(sum(int(x["case_id"])==c for x in rows)!=5 for c in expected_cases):
    raise SystemExit("not five attempts per case")
if sum(bool(x["used_for_checkpoint_selection"]) for x in rows)!=12:
    raise SystemExit("selection reuse count mismatch")
def p95(v):
    return sorted(v)[max(0,math.ceil(.95*len(v))-1)]
def agg(rs):
    dur=[float(x["capped_minutes"]) for x in rs];mean=statistics.mean(dur);sd=statistics.pstdev(dur)
    return {
      "attempts":len(rs),"strict_correct":sum(bool(x["correct"]) for x in rs),
      "accuracy_percent":100*sum(bool(x["correct"]) for x in rs)/len(rs),
      "model_hard_timeouts":sum(bool(x["timeout"]) for x in rs),
      "runner_failures":sum(bool(x.get("infrastructure_failure", x["runner_status"]!="succeeded" and not x.get("timeout"))) for x in rs),
      "model_completed_without_valid_answer":sum(
          bool(x.get("model_completed_without_valid_answer")) for x in rs
      ),
      "false_positives":sum(int(x["false_positive_count"]) for x in rs),
      "false_negatives":sum(int(x["false_negative_count"]) for x in rs),
      "runtime_minutes":{"mean":mean,"median":statistics.median(dur),"p95":p95(dur),"population_stddev":sd,"coefficient_of_variation":sd/mean if mean else None},
      "reasoning_items":sum(int(x.get("reasoning_items",0)) for x in rs),
      "reasoning_characters":sum(int(x.get("reasoning_characters",0)) for x in rs),
      "attempts_with_captured_reasoning":sum(int(x.get("reasoning_items",0))>0 for x in rs),
      "input_tokens":sum(int(x.get("input_tokens",0)) for x in rs),
      "cached_input_tokens":sum(int(x.get("cached_input_tokens",0)) for x in rs),
      "output_tokens":sum(int(x.get("output_tokens",0)) for x in rs),
    }
selection_cases={12,20,38,71,86,100}
overall=agg(rows)
out={
 "schema_version":"0804-final-agent-validation.v1","status":"completed" if overall["runner_failures"]==0 else "incomplete",
 "selected_checkpoint":sel,"case_ids":expected_cases,"repeats_per_case":5,
 "selection_reused_attempts":12,
 "overall":overall,
 "selection_cases":agg([x for x in rows if int(x["case_id"]) in selection_cases]),
 "nonselection_cases":agg([x for x in rows if int(x["case_id"]) not in selection_cases]),
 "per_case":{str(c):agg([x for x in rows if int(x["case_id"])==c]) for c in expected_cases},
 "runs":rows,
}
dest=run/"final_validation";dest.mkdir(exist_ok=True)
(dest/"validation_summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
with open(dest/"attempts.csv","w",newline="",encoding="utf-8-sig") as f:
    fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
    for r in rows:
        q=dict(r)
        for k in ("prediction","expected"):q[k]=json.dumps(q[k],ensure_ascii=False)
        w.writerow(q)
print(json.dumps(out["overall"],ensure_ascii=False))

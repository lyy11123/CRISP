"""Host CPU and both closeout queues; observation only."""
import json
from pathlib import Path
import time
import hi_monitor

root=Path(__file__).resolve().parent/"hi"
destination=root/"out/closeout_20261115/host_monitor.jsonl"
previous=hi_monitor.cpu_ticks()
time.sleep(1)
while True:
    targets=[("closeout_20261115","closeout")]
    j=root/"out/closeout_j_20261115"
    if (j/"closeout.status.json").exists():targets.append((j.name,"closeout"))
    record,previous=hi_monitor.sample(root,targets,previous)
    text=json.dumps(record)
    with destination.open("a",encoding="utf-8") as stream:stream.write(text+"\n")
    print(text,flush=True)
    if len(targets)==2 and all(b["finished"] for b in record["batches"]):break
    time.sleep(45)

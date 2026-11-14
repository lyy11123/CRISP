"""Read running H/I batch status and host CPU without importing experiment code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def cpu_ticks():
    values = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    return sum(values[:8]), values[3] + values[4]


def sample(root, targets, previous):
    current = cpu_ticks()
    total, idle = (a-b for a, b in zip(current, previous))
    memory = {row.split(':')[0]: int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()}
    batches = []
    for name, prefix in targets:
        folder = root/'out'/name
        status = json.loads((folder/(prefix+'.status.json')).read_text())
        finish_path = folder/'controller_finish.json'
        finish = json.loads(finish_path.read_text()) if finish_path.exists() else None
        if finish:
            status.update(completed=finish['completed'], active=0, pending=finish['expected']-finish['completed'], running=[])
        phases = []
        for row in sorted(status['running'], key=lambda r: -r['seconds'])[:4]:
            path = folder/'progress'/(row['id']+'.json')
            progress = json.loads(path.read_text()) if path.exists() else {}
            phases.append(dict(id=row['id'], phase=progress.get('phase'), step=progress.get('step'), seconds=row['seconds']))
        batches.append(dict(name=name, completed=status['completed'], active=status['active'], pending=status['pending'],
            failed=status['failed'], failed_attempt_count=status.get('failed_attempt_count', 0), phases=phases,
            finished=finish is not None, finish=finish))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), cpu_percent=round(100*(1-idle/total),1),
        available_gib=memory['MemAvailable']/1048576, active=sum(b['active'] for b in batches), batches=batches)
    return result, current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=45)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent/'hi'
    previous = cpu_ticks()
    time.sleep(1)
    while True:
        targets = json.loads((root/'monitor_targets.json').read_text())['batches']
        result, previous = sample(root, targets, previous)
        line = json.dumps(result, ensure_ascii=False)
        with (root/'out/combined_monitor.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
        print(line, flush=True)
        if args.once or all(b['finished'] for b in result['batches']):
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()

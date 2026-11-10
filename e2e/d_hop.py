"""The per-hop backend of e2e/fast_hop.py with a stop on a target stream (D_PLAN.md section 6, "decode 完成即停止").

The source of fast_hop.py is read and three anchored insertions are made (each anchor must occur exactly once); nothing else is
changed, so the random calls, the issue order and every event time are fast_hop's up to the stop. With stop_stream=None the
code path is fast_hop's plus one integer test per tick (checked bit for bit in D0).

stop_stream: stream id (tag[1] - 10). The run ends after the tick in which the last event of that stream retires (Phase 2 of
that tick, and the issues of that tick, are done). stop_info (dict): gets t_stop. The returned triple is then
(t_stop, pure compute, pure communication) up to t_stop.
"""
import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
import fast_hop as _fh  # noqa: E402  (found on sys.path; d_joint.py puts the main checkout's e2e first)

SOURCE = _fh.__file__
with open(SOURCE, encoding="utf-8") as _f:
    _src = _f.read()
SOURCE_SHA1 = hashlib.sha1(_src.encode("utf-8")).hexdigest()

EDITS = [
    ("                      shadow=None, shadow_log=None):\n    policy = policy or P1()\n",
     "                      shadow=None, shadow_log=None, stop_stream=None, stop_info=None):\n    policy = policy or P1()\n"),
    ("    on_issue = policy.on_issue\n",
     "    on_issue = policy.on_issue\n"
     "    stop_left = sum(1 for _t in events_dict if stream_of(_t) == stop_stream) if stop_stream is not None else -1\n"),
    ("                policy.on_finish(finished_event, current_time)\n",
     "                policy.on_finish(finished_event, current_time)\n"
     "                if stop_left > 0 and stream_of(finished_event.event_tag) == stop_stream:\n"
     "                    stop_left -= 1\n"),
    ("        if not alive and not wait_events and not issued_events:\n            break\n",
     "        if stop_left == 0:\n"
     "            if stop_info is not None:\n"
     "                stop_info[\"t_stop\"] = current_time\n"
     "            break\n"
     "        if not alive and not wait_events and not issued_events:\n            break\n"),
]
for _a, _b in EDITS:
    if _src.count(_a) != 1:
        raise RuntimeError(f"d_hop: anchor occurs {_src.count(_a)} times in {SOURCE}: {_a[:60]!r}")
    _src = _src.replace(_a, _b)
_ns = {"__name__": "d_hop_impl", "__file__": SOURCE}
exec(compile(_src, "<d_hop: fast_hop.py with stop>", "exec"), _ns)
event_driver_stop = _ns["event_driver_fast"]

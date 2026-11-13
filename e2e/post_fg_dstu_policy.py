"""Frozen-C4 and resource-local message continuation for post-F/G trials."""
import collections


def sid(event):
    return event.event_tag[1] - 10


def heads(event):
    result = {}
    for branch in event.path_list:
        if branch:
            result[branch[0]] = max(result.get(branch[0], 0), len(branch))
    return result


class MessagePolicy:
    """key is a pure query; a round's ready order is already backend-shuffled."""

    def __init__(self, rank, metadata, mode="C4", trace=None):
        self.rank = {int(s): r for s, r in rank.items()}
        self.metadata, self.mode, self.trace = metadata, mode, trace
        self.owner = {}
        self.preferred = {}
        self.candidates = {}
        self.stats = collections.Counter()
        self.ready_since = {}
        self.parent_order = {p: i for i, p in enumerate(sorted({m["parent"] for m in metadata.values()}))}

    def on_tick(self, t, ready):
        if self.mode != "MsgCont" and self.trace is None:
            return
        candidates = collections.defaultdict(list)
        for index, event in enumerate(ready):
            for link, length in heads(event).items():
                candidates[link].append((event, length, index))
                if self.trace is not None:
                    self.ready_since.setdefault((event.event_tag, link), t)
        self.candidates = candidates
        self.preferred = {}
        if self.mode == "MsgCont":
            for link, entries in candidates.items():
                streams = collections.defaultdict(list)
                for event, length, index in entries:
                    streams[sid(event)].append((event, length, index))
                for stream, group in streams.items():
                    owner = self.owner.get((link, stream))
                    if any(self.metadata[e.event_tag]["parent"] == owner for e, _, _ in group):
                        parent = owner
                    else:
                        event = min(group, key=lambda row: (-row[1], row[2]))[0]
                        parent = self.metadata[event.event_tag]["parent"]
                    self.preferred[(link, stream)] = parent

    def key(self, event, link, length, t):
        stream = sid(event)
        meta = self.metadata[event.event_tag]
        if self.mode == "C4":
            return self.rank[stream], -length
        if self.mode == "TieFIFO":
            return self.rank[stream], -length, self.parent_order[meta["parent"]], meta["index"]
        chosen = meta["parent"] == self.preferred[(link, stream)]
        return self.rank[stream], 0 if chosen else 1, meta["index"] if chosen else -length

    def on_issue(self, event, link, t):
        stream = sid(event)
        parent = self.metadata[event.event_tag]["parent"]
        previous = self.owner.get((link, stream))
        group = self.candidates.get(link, [])
        previous_ready = any(sid(e) == stream and self.metadata[e.event_tag]["parent"] == previous
                             for e, _, _ in group)
        interrupted = previous is not None and previous != parent and previous_ready
        lower_ready = any(self.rank[sid(e)] > self.rank[stream] for e, _, _ in group)
        self.stats["issues"] += 1
        self.stats["same_stream_insertion"] += int(interrupted)
        self.stats["higher_rank_issue_with_lower_ready"] += int(lower_ready)
        if self.trace is not None:
            length = heads(event)[link]
            original = (self.rank[stream], -length)
            original_min = min((self.rank[sid(e)], -ln) for e, ln, _ in group)
            same_key = sum((self.rank[sid(e)], -ln) == original for e, ln, _ in group)
            self.trace(dict(t=t, resource=link, stream=stream, parent=parent,
                            chunk=self.metadata[event.event_tag]["index"],
                            original_key=original, original_min=original_min,
                            original_equal_candidates=same_key, original_true_tie=same_key > 1,
                            previous_parent=previous, previous_parent_ready=previous_ready,
                            same_stream_insertion=interrupted, lower_rank_ready=lower_ready,
                            head_ready=self.ready_since.pop((event.event_tag, link)),
                            key=self.key(event, link, length, t)))
        # A higher-priority issue changes only its own (resource, stream) owner.
        # The lower-priority parent's ownership survives preemption. A same-flow
        # replacement takes ownership when it actually issues, never in key().
        self.owner[(link, stream)] = parent

    def on_finish(self, event, t):
        # Ownership is the last serviced parent, independently at each resource.
        # No ready child means another parent may issue; its issue replaces owner.
        # A stale owner has no preference unless a child is currently ready.
        pass

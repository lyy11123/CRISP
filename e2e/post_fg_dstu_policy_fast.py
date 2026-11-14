"""MessagePolicy with per-round candidate summaries; identical arbitration keys."""
import collections

from post_fg_dstu_policy import heads


class MessagePolicy:
    def __init__(self, rank, metadata, mode="C4", trace=None):
        self.rank = {int(s): r for s, r in rank.items()}
        self.metadata, self.mode, self.trace = metadata, mode, trace
        self.owner = {}
        self.preferred = {}
        self.candidates = {}
        self.stats = collections.Counter()
        self.ready_since = {}
        self.parent_order = {p: i for i, p in enumerate(sorted({m["parent"] for m in metadata.values()}))}
        self.info = {tag: (tag[1]-10, m["parent"], m["index"], self.rank[tag[1]-10])
                     for tag, m in metadata.items()}
        self.summaries = {}
        self.head_cache = {}
        self.changed_heads = set()

    def on_tick(self, t, ready):
        if self.mode != "MsgCont" and self.trace is None:
            return
        candidates = {}
        best = {}
        owner_ready = set()
        traced = self.trace is not None
        continuity = self.mode == "MsgCont"
        for index, event in enumerate(ready):
            tag = event.event_tag
            stream, parent, _, rank = self.info[tag]
            if tag not in self.head_cache or tag in self.changed_heads:
                self.head_cache[tag] = heads(event)
                self.changed_heads.discard(tag)
            for link, length in self.head_cache[tag].items():
                if link not in candidates:
                    candidates[link] = [(tag, length)]
                else:
                    candidates[link].append((tag, length))
                if continuity:
                    group = (link, stream)
                    if group not in best or length > best[group][0]:
                        best[group] = (length, parent)
                    if parent == self.owner.get(group):
                        owner_ready.add(group)
                if traced:
                    self.ready_since.setdefault((tag, link), t)
        self.candidates = candidates
        self.summaries = {}
        self.preferred = {}
        if continuity:
            for group, (_, parent) in best.items():
                self.preferred[group] = self.owner[group] if group in owner_ready else parent

    def summary(self, link):
        """Only resources that issue need the diagnostic candidate summary."""
        if link not in self.summaries:
            parents, keys = set(), {}
            highest = None
            minimum = None
            for tag, length in self.candidates.get(link, ()):
                stream, parent, _, rank = self.info[tag]
                parents.add((stream, parent))
                if highest is None or rank > highest:
                    highest = rank
                if self.trace is not None:
                    key = (rank, -length)
                    keys[key] = keys.get(key, 0)+1
                    if minimum is None or key < minimum:
                        minimum = key
            self.summaries[link] = parents, highest, keys, minimum
        return self.summaries[link]

    def key(self, event, link, length, t):
        stream, parent, index, rank = self.info[event.event_tag]
        if self.mode == "C4":
            return rank, -length
        if self.mode == "TieFIFO":
            return rank, -length, self.parent_order[parent], index
        chosen = parent == self.preferred[(link, stream)]
        return rank, 0 if chosen else 1, index if chosen else -length

    def on_issue(self, event, link, t):
        tag = event.event_tag
        stream, parent, index, rank = self.info[tag]
        group = (link, stream)
        previous = self.owner.get(group)
        parents, highest, keys, minimum = self.summary(link)
        previous_ready = (stream, previous) in parents
        interrupted = previous is not None and previous != parent and previous_ready
        lower_ready = highest is not None and highest > rank
        self.stats["issues"] += 1
        self.stats["same_stream_insertion"] += int(interrupted)
        self.stats["higher_rank_issue_with_lower_ready"] += int(lower_ready)
        if self.trace is not None:
            # Other resources may already have advanced this event in this round.
            length = heads(event)[link]
            original = (rank, -length)
            same_key = keys.get(original, 0)
            self.trace(dict(t=t, resource=link, stream=stream, parent=parent,
                            chunk=index, original_key=original, original_min=minimum,
                            original_equal_candidates=same_key, original_true_tie=same_key > 1,
                            previous_parent=previous, previous_parent_ready=previous_ready,
                            same_stream_insertion=interrupted, lower_rank_ready=lower_ready,
                            head_ready=self.ready_since.pop((tag, link)),
                            key=self.key(event, link, length, t)))
        self.owner[group] = parent
        # fast_hop removes the issued head immediately after this callback.
        self.changed_heads.add(tag)

    def on_finish(self, event, t):
        self.head_cache.pop(event.event_tag, None)
        self.changed_heads.discard(event.event_tag)

"""Small deterministic scheduling and chunk semantics checks; no large simulation."""
import copy
import json
import pickle
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from post_fg_dstu_policy import MessagePolicy
from post_fg_dstu import ROOT, write_json
import fast_hop


class Event:
    def __init__(self, tag, paths, size=3):
        self.event_tag=tag;self.event_type="communication";self.path_list=copy.deepcopy(paths)
        self.comm_bytes=size;self.dependency_set=set();self.issue_set=set()
        self.start_time=None;self.end_time=0


class Link:
    def __init__(self):self.work_flag=False;self.work_endtime=0;self.work_record=[]
    def working_time(self,data_bytes):return data_bytes+1


class Fixed:
    def __init__(self,rank):self.rank=rank
    def key(self,e,link,length,t):return self.rank[e.event_tag[1]-10],-length
    def on_tick(self,t,ready):pass
    def on_issue(self,e,link,t):pass
    def on_finish(self,e,t):pass


def main():
    checks=[];L=(0,1);Q=(1,2)
    tags=[(1,10,0,1),(1,10,0,1000001),(1,10,0,2),(1,10,0,1000002),(1,11,0,3)]
    a0,a1,b0,b1,d=[Event(t,[[L]]) for t in tags]
    metadata={e.event_tag:dict(parent=tags[0] if e in (a0,a1) else tags[2] if e in (b0,b1) else tags[4],
                              index=int(e in (a1,b1)),count=2 if e is not d else 1) for e in (a0,a1,b0,b1,d)}
    p=MessagePolicy({0:1,1:0},metadata,"MsgCont")
    b0.path_list=[[L,Q]];p.on_tick(0,[a0,b0,a1])
    assert min((p.key(e,L,len(e.path_list[0]),0),i,e) for i,e in enumerate((a0,b0,a1)))[2] is b0
    checks.append("initial parent follows original longest remaining path and shuffled tie order")
    p.on_issue(b0,L,0);a0.path_list=[[L,Q]];p.on_tick(4,[a0,b1,a1])
    assert p.key(b1,L,1,4)<p.key(a0,L,2,4)
    checks.append("ready owner continues over another same-flow longer path")
    p.on_tick(4,[a0,b1,d]);assert p.key(d,L,1,4)<p.key(b1,L,1,4)
    p.on_issue(d,L,4);assert p.owner[(L,0)]==tags[2]
    p.on_tick(8,[a0,b1]);assert p.key(b1,L,1,8)<p.key(a0,L,2,8)
    checks.append("higher-priority stream preempts; lower-flow ownership resumes")
    p.on_tick(8,[a0,a1]);assert min((p.key(e,L,len(e.path_list[0]),8),i,e) for i,e in enumerate((a0,a1)))[2] is a0
    p.on_issue(a0,L,8);assert p.owner[(L,0)]==tags[0]
    p.on_tick(12,[a1,a0]);assert p.key(a0,L,2,12)<p.key(a1,L,1,12)
    checks.append("unready owner does not idle the resource; chosen parent uses explicit chunk index")
    state=pickle.dumps(p.__dict__);rng=random.getstate()
    for _ in range(5):p.key(a1,L,1,12)
    assert pickle.dumps(p.__dict__)==state and random.getstate()==rng
    checks.append("repeated key queries change neither policy state nor RNG")
    trace=[];q=MessagePolicy({0:1,1:0},metadata,"C4",trace.append)
    q.on_tick(0,[a0,a1]);q.on_issue(a0,L,0)
    assert trace[0]["original_true_tie"] is False
    a1.path_list=[[L,Q]];q.on_tick(4,[a0,a1]);q.on_issue(a1,L,4)
    assert trace[1]["original_true_tie"] is True
    checks.append("resource trace distinguishes original-key ties from different path lengths")

    events={tags[0]:Event(tags[0],[[L,Q]]),tags[2]:Event(tags[2],[[L]]),tags[4]:Event(tags[4],[[L,Q]])}
    meta={t:dict(parent=t,index=0,count=1) for t in events}
    def run(policy):
        ev=copy.deepcopy(events);hp=SimpleNamespace(links_dict={L:Link(),Q:Link()},modules_dict={})
        random.seed(7);result=fast_hop.event_driver_fast(ev,hp,release={0:0,1:2},policy=policy)
        return result,{t:(e.start_time,e.end_time) for t,e in ev.items()},{k:l.work_record for k,l in hp.links_dict.items()},random.getstate()
    original=run(Fixed({0:1,1:0}));ours=run(MessagePolicy({0:1,1:0},meta))
    assert original==ours
    checks.append("new C4 policy equals old C4 total/event/resource records and RNG on small backend graph")
    single={tags[0]:Event(tags[0],[[L,Q]])}
    hp=SimpleNamespace(links_dict={L:Link(),Q:Link()},modules_dict={})
    fast_hop.event_driver_fast(single,hp,policy=MessagePolicy({0:0},{tags[0]:dict(parent=tags[0],index=0,count=1)},"MsgCont"))
    assert hp.links_dict[L].work_record[0][0]==hp.links_dict[Q].work_record[0][0]==0
    checks.append("same tick multi-hop issue remains unchanged")
    # Final block completion releases a dependent communication consumer.
    ctag=(1,10,0,9);consumer=Event(ctag,[[Q]],1)
    ev={tags[0]:Event(tags[0],[[L]],2),tags[1]:Event(tags[1],[[L]],2),ctag:consumer}
    for t in (tags[0],tags[1]):ev[t].issue_set={ctag}
    consumer.dependency_set={tags[0],tags[1]}
    m={t:dict(parent=tags[0] if t!=ctag else ctag,index=i if t!=ctag else 0,count=2 if t!=ctag else 1) for i,t in enumerate(ev)}
    hp=SimpleNamespace(links_dict={L:Link(),Q:Link()},modules_dict={})
    fast_hop.event_driver_fast(ev,hp,policy=MessagePolicy({0:0},m,"MsgCont"))
    assert hp.links_dict[Q].work_record[0][0]==max(ev[t].end_time for t in (tags[0],tags[1]))
    checks.append("consumer waits for all chunks without serial dependencies between chunks")
    result=dict(status="passed",checks=checks,check_count=len(checks),scope="synthetic event graphs only")
    write_json(ROOT/"e2e/post_fg/evidence/dstu_policy_checks.json",result)
    print(json.dumps(result))


if __name__=="__main__":main()

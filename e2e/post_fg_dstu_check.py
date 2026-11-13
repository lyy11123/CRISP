"""Small deterministic scheduling and chunk semantics checks; no large simulation."""
import copy
import gzip
import json
import pickle
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

from post_fg_dstu_policy import MessagePolicy
from post_fg_dstu import ROOT, write_json
import post_fg_dstu as runner
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


def real_alias_checks():
    """Real graph identities, plus a mocked result handoff; no backend calls."""
    manifest=runner.read_json(ROOT/"e2e/post_fg/manifest/dstu_health.json")
    checks=[];out=ROOT/"e2e/post_fg/out/local_check"
    for hw in ("wsc","face"):
        spec=next(s for s in manifest["jobs"] if s["hardware_profile"]==hw and s.get("alias_job") and s["seed"]==1)
        hspec=next(s for s in manifest["jobs"] if s["job_id"]==spec["alias_job"])
        d,hp,g,streams=runner.graph_inputs(spec,out)
        h0,_,_,hstats=runner.transform(copy.deepcopy(streams),hp,"H0",spec["rank"])
        filtered_h0={t:e for t,e in h0.items() if t[1]==11}
        expected=runner.evaluation_identity(hspec,filtered_h0,hp,{1:0})
        # The time below is a sentinel fixture, not a simulation measurement.
        other=dict(status="ok",spec=hspec,evaluation_identity=expected,completion={"1":37},
                   first_start={"1":0},kv_window=None,release={"1":0},simulated_total=37,
                   unhidden_compute=11,unhidden_communication=9,policy_counters={"issues":2},
                   candidate_counters_available=False,audit=dict(n=2,bad=0),simulation_seconds=1.25)
        with patch.object(runner,"read_json",return_value=other),patch.object(runner,"sha",return_value="fixture-result-digest"),\
             patch.object(runner,"graph_inputs",side_effect=lambda *args:(d,hp,g,copy.deepcopy(streams))):
            result=runner.batch_alias(spec,out)
            assert result["evaluation_identity"]==expected and result["completion"]=={"1":37}
            assert result["chunk_stats"]["selected_messages"]==128 and hstats["selected_messages"]==0
            assert result["backend_calls"]==0 and result["simulation_seconds"]==0
            with gzip.open(result["message_transform"],"rt",encoding="utf-8") as handle:stored=json.load(handle)
            assert stored["arm"]=="S256" and stored["stats"]["blocks_added"]==256
            assert any(m["tag"][1]==10 and m["count"]>1 for m in stored["messages"])
            assert all(m["count"]==1 for m in stored["messages"] if m["tag"][1]==11)
            other["evaluation_identity"]=dict(expected,seed=2)
            try:runner.batch_alias(spec,out)
            except AssertionError:pass
            else:raise AssertionError("different backend seed was reused")
        checks.append(dict(hardware=hw,D_graph_sha256=expected["graph_sha256"],
                           filtered_graph_policy_platform_release_seed_match=True,
                           S256_joint_mask_retained=True,H0_metadata_not_copied=True,
                           mismatched_seed_rejected=True,backend_calls=0,result_transfer_fixture_only=True))
    result=dict(status="passed",checks=checks,logical_units=180,historical_reuses=60,new_batch_aliases=6,
                new_backend_calls=114,smoke_backend_calls=15,scope="Real saved graph identity and mocked result transfer; no simulation")
    write_json(ROOT/"e2e/post_fg/evidence/dstu_alias_checks.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    if "--real-alias" in sys.argv:
        real_alias_checks()
    else:main()

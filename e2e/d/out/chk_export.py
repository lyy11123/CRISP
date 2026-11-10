import gzip,pickle,glob,sys,os
sys.path.insert(0,os.getcwd())
import d_joint as d
c=d._sim_setup()
for p in sorted(glob.glob('d/out/d0/export/*.pkl.gz')):
    ev=pickle.loads(pickle.load(gzip.open(p))['d_export']['events_blob'])
    comm=[e for e in ev.values() if e.event_type=='communication']
    empty=sum(1 for e in comm if not e.path_list or any(len(b)==0 for b in e.path_list))
    st=sum(1 for e in ev.values() if getattr(e,'start_time',None) not in (None,0))
    en=sum(1 for e in ev.values() if getattr(e,'end_time',None) is not None)
    nodep=sum(1 for e in ev.values() if not e.dependency_set)
    print(os.path.basename(p)[:34], len(ev), len(comm), 'empty_paths',empty,'start_set',st,'end_set',en,'sources',nodep)

# Loaded only when cpcm_v2.py launches `python <task>.py` with this folder on PYTHONPATH. Same as
# e3_warmstart/_inject/sitecustomize.py (E3's capture and inject modes, unchanged), except that in inject mode
# e3_warmstart._inject_eval is replaced by cpcm_v2.inject_eval (compile the graph for CPCM, run backend seeds).
import os
import sys

if os.environ.get("E3_MODE"):
    import importlib.machinery

    _CAPTURE = os.environ["E3_MODE"] == "capture"
    _WANTED = {"event_driver"} if _CAPTURE else \
        {"event_driver", "read_cfg", "Pre_Mapping", "Stream_Mapping", "add_communication", "WAMIS_HD"}

    def _e3():
        if "e3_warmstart" not in sys.modules:
            sys.path.insert(0, os.environ["E3_HOME"])
            import e3_warmstart
            sys.path.pop(0)
            if not _CAPTURE:
                sys.path.insert(0, os.environ["CPCM_HOME"])
                import cpcm_v2
                sys.path.pop(0)
                e3_warmstart._inject_eval = cpcm_v2.inject_eval
        return sys.modules["e3_warmstart"]

    def _patch(name, module):
        if not _CAPTURE:
            if name == "event_driver":
                module._orig_event_driver = module.event_driver   # for cpcm_v4's trajectory sampling
            _e3().patch_module(name, module)
            if name == "Stream_Mapping" and os.environ.get("CPCM_V4_SAMPLE"):
                # cpcm_v4 (V4 fault tau): sample the post-fault SA trajectory with cpcm_v3.make_anneal; broadcast
                # through E3's wrapped add_broadcast (route restore, as E3's own W evaluation), backend through
                # the original event_driver
                sys.path.insert(0, os.environ["CPCM_HOME"])
                import cpcm_v3
                sys.path.pop(0)
                lab, part, nparts, every, sink, seeds = os.environ["CPCM_V4_SAMPLE"].split("|")
                module.StreamOptimization.anneal = cpcm_v3.make_anneal(
                    lab, int(part), int(nparts), int(every), [int(x) for x in seeds.split()], sink, False,
                    lambda **kw: sys.modules["add_communication"].add_broadcast(**kw),
                    lambda **kw: sys.modules["event_driver"]._orig_event_driver(**kw))
            return
        orig, calls = module.event_driver, [0]

        def event_driver(events_dict, hardware_platform):
            calls[0] += 1
            if calls[0] == 1:  # initial mapping: left untouched
                return orig(events_dict=events_dict, hardware_platform=hardware_platform)
            g = sys._getframe(1).f_globals
            return _e3().capture_final(orig, events_dict, hardware_platform, calls[0], g)

        module.event_driver = event_driver

    class _Finder:
        def find_spec(self, name, path=None, target=None):
            if name not in _WANTED:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None:
                return None
            loader = spec.loader

            class _Loader(type(loader)):
                def exec_module(self_, module):
                    loader.exec_module(module)
                    _patch(name, module)

            spec.loader = _Loader(loader.name, loader.path)
            _WANTED.discard(name)
            if not _WANTED:
                sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, _Finder())

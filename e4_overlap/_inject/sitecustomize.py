# Loaded only when e4_overlap.py launches `python <task>.py` with this folder on PYTHONPATH.
# capture: same approach as E2/E3, nothing is allocated before the template's final (2nd) backend call beyond
#   this import hook; e4_overlap is imported only at that call.
# inject (step 4, faults): E3's in-memory patches (e3_warmstart.patch_module: [failures] from the spec, mapping
#   injection, R0 relocation, route restoration, the WAMIS_HD xy_multicast_path fix) on read_cfg, Pre_Mapping,
#   Stream_Mapping, add_communication and WAMIS_HD, driven by E3_SPEC; the backend's 1st call is skipped as in
#   E3, and at the 2nd call E4's capture_final pickles the repaired event objects.
import os
import sys

_MODE = os.environ.get("E4_MODE")
if _MODE in ("capture", "inject"):
    import importlib.machinery

    _WANTED = {"event_driver"} if _MODE == "capture" else \
        {"event_driver", "read_cfg", "Pre_Mapping", "Stream_Mapping", "add_communication", "WAMIS_HD"}

    def _e3():
        sys.path.insert(0, os.environ["E3_HOME"])
        import e3_warmstart
        sys.path.pop(0)
        return e3_warmstart

    def _patch(name, module):
        if name != "event_driver":
            _e3().patch_module(name, module)
            return
        orig, calls = module.event_driver, [0]

        def event_driver(events_dict, hardware_platform):
            calls[0] += 1
            if calls[0] == 1:
                if _MODE == "inject":  # initial mapping: the template discards the result (as E3)
                    return (0, 0, 0)
                return orig(events_dict=events_dict, hardware_platform=hardware_platform)
            g = sys._getframe(1).f_globals  # the template's module globals (beha_dict, data_dict, ...)
            sys.path.insert(0, os.environ["E4_HOME"])
            import e4_overlap
            sys.path.pop(0)
            return e4_overlap.capture_final(orig, events_dict, hardware_platform, calls[0], g)

        module.event_driver = event_driver

    class _E4Finder:
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

    sys.meta_path.insert(0, _E4Finder())

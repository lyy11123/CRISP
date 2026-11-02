# Loaded only when e3_warmstart.py launches `python <task>.py` with this folder on PYTHONPATH.
# Kept minimal: BusyBarn's results depend on object addresses, so in "capture" mode nothing is allocated
# before the template's final backend call beyond this import hook (same as E2); e3_warmstart is imported
# only at the 2nd event_driver call. In "inject" mode read_cfg / Pre_Mapping / Stream_Mapping /
# add_communication are patched too, and e3_warmstart is imported as soon as the first of them loads.
import os
import sys

if os.environ.get("E3_MODE"):
    import importlib.machinery

    _CAPTURE = os.environ["E3_MODE"] == "capture"
    _WANTED = {"event_driver"} if _CAPTURE else \
        {"event_driver", "read_cfg", "Pre_Mapping", "Stream_Mapping", "add_communication", "WAMIS_HD"}

    def _e3():
        sys.path.insert(0, os.environ["E3_HOME"])
        import e3_warmstart
        sys.path.pop(0)
        return e3_warmstart

    def _patch(name, module):
        if not _CAPTURE:
            _e3().patch_module(name, module)
            return
        orig, calls = module.event_driver, [0]

        def event_driver(events_dict, hardware_platform):
            calls[0] += 1
            if calls[0] == 1:  # initial mapping: left untouched
                return orig(events_dict=events_dict, hardware_platform=hardware_platform)
            g = sys._getframe(1).f_globals  # the template's module globals (beha_dict, data_dict, ...)
            return _e3().capture_final(orig, events_dict, hardware_platform, calls[0], g)

        module.event_driver = event_driver

    class _E3Finder:
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

    sys.meta_path.insert(0, _E3Finder())

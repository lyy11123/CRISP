# Loaded only when cpcm_sa.py check3 launches `python <task>.py` with this folder on PYTHONPATH. E3's inject mode
# (e3_warmstart/_inject/sitecustomize.py, unchanged: mapping, sources, routes, comm tags and event order restored,
# R0 = no SA), plus cpcm_sa.check3_child_hook wrapped around E3's stream_mapping: CPCM on the injected
# pre-broadcast state, then exit.
import os
import sys

if os.environ.get("E3_MODE") == "inject":
    import importlib.machinery

    _WANTED = {"event_driver", "read_cfg", "Pre_Mapping", "Stream_Mapping", "add_communication", "WAMIS_HD"}

    def _e3():
        if "e3_warmstart" not in sys.modules:
            sys.path.insert(0, os.environ["E3_HOME"])
            import e3_warmstart  # noqa: F401
            sys.path.pop(0)
        return sys.modules["e3_warmstart"]

    def _patch(name, module):
        _e3().patch_module(name, module)
        if name == "Stream_Mapping":
            sys.path.insert(0, os.environ["CPCM_SA_HOME"])
            import cpcm_sa
            sys.path.pop(0)
            cpcm_sa.check3_child_hook()

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

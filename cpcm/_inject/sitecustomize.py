# Loaded only when cpcm_v1.py launches `python <task>.py` with this folder on PYTHONPATH (same pattern as
# e2_attribution/_inject). Nothing is allocated here beyond the import hook; the capture code is imported at the
# template's 2nd (final) event_driver call.
import os
import sys

if os.environ.get("CPCM_SINK"):
    import importlib.machinery

    class _CPCMFinder:
        def find_spec(self, name, path=None, target=None):
            if name != "event_driver":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None:
                return None
            loader = spec.loader

            class _Loader(type(loader)):
                def exec_module(self_, module):
                    loader.exec_module(module)
                    orig, calls = module.event_driver, [0]

                    def event_driver(events_dict, hardware_platform):
                        calls[0] += 1
                        if calls[0] == 1:  # initial mapping: left untouched
                            return orig(events_dict=events_dict, hardware_platform=hardware_platform)
                        sys.path.insert(0, os.environ["CPCM_HOME"])
                        import cpcm_v1
                        sys.path.pop(0)
                        return cpcm_v1.child_capture(orig, events_dict, hardware_platform, calls[0])

                    module.event_driver = event_driver

            spec.loader = _Loader(loader.name, loader.path)
            sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, _CPCMFinder())

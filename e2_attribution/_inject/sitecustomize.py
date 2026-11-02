# Loaded only when e2_attribution.py launches `python <task>.py` with this folder on PYTHONPATH.
# Kept minimal on purpose: BusyBarn's results depend on object addresses, so nothing is allocated here
# beyond registering an import hook; the capture code is imported once the script imports event_driver.
import os
import sys

if os.environ.get("E2_SINK"):
    import importlib.machinery

    class _E2Finder:
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
                        if calls[0] == 1:  # initial mapping: not analysed, left untouched
                            return orig(events_dict=events_dict, hardware_platform=hardware_platform)
                        sys.path.insert(0, os.environ["E2_HOME"])
                        import e2_attribution
                        sys.path.pop(0)
                        return e2_attribution.child_capture(orig, events_dict, hardware_platform, calls[0])

                    module.event_driver = event_driver

            spec.loader = _Loader(loader.name, loader.path)
            sys.meta_path.remove(self)
            return spec

    sys.meta_path.insert(0, _E2Finder())

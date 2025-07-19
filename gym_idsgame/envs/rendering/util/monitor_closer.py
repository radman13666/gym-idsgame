from contextlib import ExitStack
from typing import Callable, Dict

class MonitorCloser:
    def __init__(self):
        self._stack = ExitStack()
        self._closeables: Dict[int, Callable[[], None]] = {}

    def register(self, obj):
        """Register an object with a `close()` method."""
        def _closer():
            try:
                obj.close()
            except Exception:
                pass  # Optional: log exception

        obj_id = id(obj)
        self._closeables[obj_id] = _closer
        self._stack.callback(_closer)
        return obj_id

    def unregister(self, obj_id):
        """Unregister a previously registered object."""
        if obj_id in self._closeables:
            del self._closeables[obj_id]
        # Note: Can't remove from ExitStack; just skip calling it later

    def close(self):
        self._stack.close()
        self._closeables.clear()

    @property
    def closeables(self):
        """Return currently registered closeables."""
        return self._closeables

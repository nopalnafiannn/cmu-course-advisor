"""
Monkey patches for fixing compatibility issues
"""
import types

# Try to patch torch._classes to fix the issue with Streamlit's file watcher
try:
    import torch._classes
    
    # Create a stub for __path__._path that won't cause errors
    class StubPath:
        _path = []
    
    # Create a patched version of the module
    patched_classes = types.ModuleType('torch._classes')
    for attr_name in dir(torch._classes):
        if not attr_name.startswith('__'):
            setattr(patched_classes, attr_name, getattr(torch._classes, attr_name))
    
    # Add our stub path
    patched_classes.__path__ = StubPath()
    
    # Replace the original module in sys.modules
    import sys
    sys.modules['torch._classes'] = patched_classes
    
    print("Applied torch._classes patch for Streamlit compatibility")
except (ImportError, AttributeError) as e:
    print(f"Could not apply torch._classes patch: {e}")
# Patch streamlit local_sources_watcher to skip scanning torch modules
try:
    import streamlit.watcher.local_sources_watcher as _lsw
    _orig_get_module_paths = _lsw.get_module_paths
    def _safe_get_module_paths(module):
        if getattr(module, "__name__", "").startswith("torch"):
            return set()
        try:
            return _orig_get_module_paths(module)
        except Exception:
            return set()
    _lsw.get_module_paths = _safe_get_module_paths
    print("Patched streamlit local_sources_watcher.get_module_paths to skip torch modules")
except Exception:
    pass
import sys
import os

# Ensure we have the right directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Patch to handle PyTorch custom classes issue with Streamlit
import torch._classes
orig_getattr = torch._classes.__getattr__

def patched_getattr(self, attr):
    # Skip the problematic path query
    if attr == '__path__':
        return type('StubPath', (), {'_path': []})
    return orig_getattr(self, attr)

# Apply the patch
torch._classes.__getattr__ = patched_getattr

# Now import and run your Streamlit app
import streamlit_app
"""
Test bootstrap.

mrrm.py is a Streamlit script: the dashboard UI runs at module top level
(`st.set_page_config(...)` onward) and would execute on import. To unit-test
the pure functions we load only the code *above* that UI block into an
importable module named ``mrrm``.

External libraries that are irrelevant to the core logic (streamlit, yfinance,
plotly) are stubbed. hmmlearn is left to the real import: if it is installed,
the HMM path is exercised; otherwise HMM tests skip themselves.
"""

import sys
import types
from pathlib import Path

# --- stub external dependencies ------------------------------------------
_st = types.ModuleType("streamlit")
_st.cache_data = lambda *a, **k: (lambda fn: fn)  # passthrough decorator
sys.modules["streamlit"] = _st

sys.modules["yfinance"] = types.ModuleType("yfinance")

_plotly = types.ModuleType("plotly")
_go = types.ModuleType("plotly.graph_objects")
_go.Figure = object
sys.modules["plotly"] = _plotly
sys.modules["plotly.graph_objects"] = _go

# hmmlearn intentionally NOT stubbed (real import drives _HMM_AVAILABLE).

# --- load function-only portion of mrrm.py -------------------------------
_SRC = Path(__file__).resolve().parent.parent / "mrrm.py"
_text = _SRC.read_text()

_marker = "\nst.set_page_config"
_cut = _text.index(_marker)
_functions_only = _text[:_cut]

mrrm = types.ModuleType("mrrm")
mrrm.__file__ = str(_SRC)
exec(compile(_functions_only, str(_SRC), "exec"), mrrm.__dict__)
sys.modules["mrrm"] = mrrm

"""Put the repo root on sys.path so tests can `import predkit.*` directly.

Deliberately adds nothing else. The unit tests must keep passing with no
network, no credentials, no `websockets` and no `cryptography`: those are
imported lazily inside the functions that need them, and a test that needs
one of them is no longer a unit test and says so with `importorskip`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

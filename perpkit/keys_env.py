"""BloFin API credentials, from environment variables only.

Order paths need a key; recording and every public market-data read do not.
The three values are read from the process environment at the moment a
client is built, passed straight to the SDK, and never printed, logged,
written to disk or accepted as command-line arguments (a CLI argument lands
in shell history and in the process list).

Use a trading-only key without withdrawal permission, and a demo key until
you have run a tool end to end on demo.
"""

from __future__ import annotations

import os
from typing import Tuple

ENV_NAMES = ("BLOFIN_API_KEY", "BLOFIN_API_SECRET", "BLOFIN_API_PASSPHRASE")


def blofin_credentials() -> Tuple[str, str, str]:
    """(key, secret, passphrase), or SystemExit naming every missing variable."""
    values = [os.environ.get(name, "") for name in ENV_NAMES]
    missing = [name for name, value in zip(ENV_NAMES, values) if not value]
    if missing:
        raise SystemExit(
            "Missing environment variable(s): " + ", ".join(missing) + ".\n"
            "Export them in the shell that runs this tool. They are read only "
            "from the environment and never logged.")
    return values[0], values[1], values[2]

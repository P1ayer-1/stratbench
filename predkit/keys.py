"""Non-custodial keys: encrypted on the user's machine, decrypted in-process
at signing time, never uploaded, never logged.

    python -m predkit.keys add kalshi --key-id <id> --pem-file kalshi.pem
    python -m predkit.keys add polymarket              # prompts for the private key
    python -m predkit.keys list
    python -m predkit.keys remove kalshi

The passphrase comes from `PREDKIT_PASSPHRASE` or a prompt. A secret is
never taken as a command-line argument (it would land in shell history);
it is read from `--pem-file`, from the environment variable named by
`--secret-env`, or from a hidden prompt.

Storage is one JSON file, `~/.predkit/keys.json` by default, with one entry
per name: a random salt, and the secret under Fernet (AES-128-CBC + HMAC,
from `cryptography`) with a key derived from the user's passphrase by
PBKDF2-HMAC-SHA256 at 600,000 iterations. A wrong passphrase fails the HMAC
and raises; nothing is ever half-decrypted.

What this deliberately does not do: request withdrawal permissions of any
kind (neither venue's trading key can move funds off the venue), keep the
passphrase anywhere, or hand a decrypted secret to anything but the signer
that asked for it. `load` returns bytes for the caller to build a signer
with and drop.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PATH = Path.home() / ".predkit" / "keys.json"
ITERATIONS = 600_000


def _fernet(passphrase: str, salt: bytes):
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8"))))


class Keystore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_PATH

    def _read(self) -> Dict[str, Dict[str, str]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, entries: Dict[str, Dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def names(self) -> list:
        return sorted(self._read())

    def store(self, name: str, secret: bytes, passphrase: str, *, meta: Optional[Dict[str, str]] = None) -> None:
        if not passphrase:
            raise ValueError("a passphrase is required; keys are never stored in the clear")
        salt = os.urandom(16)
        token = _fernet(passphrase, salt).encrypt(secret)
        entries = self._read()
        entries[name] = {"salt": base64.b64encode(salt).decode(), "token": token.decode(),
                         **{f"meta_{k}": v for k, v in (meta or {}).items()}}
        self._write(entries)

    def load(self, name: str, passphrase: str) -> bytes:
        from cryptography.fernet import InvalidToken

        entries = self._read()
        if name not in entries:
            raise KeyError(f"no key named {name!r} in {self.path}; known: {sorted(entries)}")
        entry = entries[name]
        try:
            return _fernet(passphrase, base64.b64decode(entry["salt"])).decrypt(entry["token"].encode())
        except InvalidToken:
            raise PermissionError(f"wrong passphrase for {name!r}") from None

    def meta(self, name: str) -> Dict[str, str]:
        entry = self._read().get(name, {})
        return {k[5:]: v for k, v in entry.items() if k.startswith("meta_")}

    def remove(self, name: str) -> None:
        entries = self._read()
        entries.pop(name, None)
        self._write(entries)


def _passphrase(confirm: bool) -> str:
    import getpass

    value = os.getenv("PREDKIT_PASSPHRASE")
    if value:
        return value
    value = getpass.getpass("keystore passphrase: ")
    if confirm and getpass.getpass("again: ") != value:
        raise SystemExit("passphrases differ")
    return value


def main(argv: Optional[list] = None) -> int:
    import argparse
    import getpass

    parser = argparse.ArgumentParser(description="Manage encrypted venue keys on this machine.")
    parser.add_argument("--keystore", default=None, help=f"default {DEFAULT_PATH}")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="encrypt and store a key under a name")
    add.add_argument("name", help="e.g. kalshi or polymarket; the runner's --key-name")
    add.add_argument("--key-id", default="", help="Kalshi API key id (stored as metadata, not secret)")
    add.add_argument("--pem-file", default=None, help="read the secret from this file (Kalshi RSA key)")
    add.add_argument("--secret-env", default=None, help="read the secret from this environment variable")
    sub.add_parser("list", help="names only; nothing is decrypted")
    rm = sub.add_parser("remove", help="delete a stored key")
    rm.add_argument("name")
    args = parser.parse_args(argv)
    store = Keystore(Path(args.keystore) if args.keystore else None)
    if args.command == "list":
        for name in store.names():
            print(name)
        return 0
    if args.command == "remove":
        store.remove(args.name)
        return 0
    if args.pem_file:
        secret = Path(args.pem_file).read_bytes()
    elif args.secret_env:
        value = os.getenv(args.secret_env)
        if not value:
            raise SystemExit(f"environment variable {args.secret_env} is empty or unset")
        secret = value.encode("utf-8")
    else:
        secret = getpass.getpass("secret (hidden): ").encode("utf-8")
    if not secret.strip():
        raise SystemExit("empty secret; nothing stored")
    meta = {"key_id": args.key_id} if args.key_id else None
    store.store(args.name, secret, _passphrase(confirm=True), meta=meta)
    print(f"stored {args.name!r} in {store.path}")
    return 0


__all__ = ["DEFAULT_PATH", "ITERATIONS", "Keystore", "main"]


if __name__ == "__main__":
    raise SystemExit(main())

"""Manage ReelSync secrets in the OS credential manager.

On Windows, secrets are stored in Credential Manager, encrypted by DPAPI and
bound to the current Windows user. Secrets are resolved at startup in this
order: OS credential manager -> environment variable -> config.toml.

Usage:
    python scripts/manage_secrets.py set OPENAI_API_KEY
    python scripts/manage_secrets.py get OPENAI_API_KEY
    python scripts/manage_secrets.py delete OPENAI_API_KEY
    python scripts/manage_secrets.py list
"""

from __future__ import annotations

import argparse
import getpass
import sys

sys.path.insert(0, __file__.rsplit("scripts", 1)[0])

from app.utils.secrets import (  # noqa: E402
    delete_secret,
    get_secret,
    has_keyring,
    set_secret,
)

# Environment variable names accepted for credential storage.
_KNOWN_SECRETS = {
    "PEXELS_API_KEY",
    "PIXABAY_API_KEY",
    "COVERR_API_KEY",
    "CUSTOM_API_KEY",
    "TWELVELABS_API_KEY",
    "SONILO_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "QWEN_API_KEY",
    "AZURE_API_KEY",
    "VOLCENGINE_API_KEY",
    "GROK_API_KEY",
    "MINIMAX_API_KEY",
    "MIMO_API_KEY",
    "CLOUDFLARE_API_KEY",
    "MODELSCOPE_API_KEY",
    "AIHUBMIX_API_KEY",
    "AIMLAPI_API_KEY",
    "EVOLINK_API_KEY",
    "ONEAPI_API_KEY",
    "GROQ_API_KEY",
    "POLLINATIONS_API_KEY",
    "UPLOAD_POST_API_KEY",
    "UPLOAD_POST_USERNAME",
    "AZURE_SPEECH_KEY",
    "SILICONFLOW_API_KEY",
    "MINIMAX_TTS_API_KEY",
    "ELEVENLABS_API_KEY",
    "CHATTERBOX_API_KEY",
    "OPENALEX_API_KEY",
    "NASA_API_KEY",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_secrets",
        description=(
            "Store and retrieve ReelSync API keys in the OS credential "
            "manager (Windows Credential Manager / macOS Keychain / libsecret)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_p = subparsers.add_parser("set", help="store a secret (prompts on terminal)")
    set_p.add_argument("name", choices=sorted(_KNOWN_SECRETS), help="env-var name of the secret")
    set_p.add_argument(
        "--value",
        default=None,
        help="optional value; if omitted the value is read from a hidden prompt",
    )

    get_p = subparsers.add_parser("get", help="print a stored secret to stdout")
    get_p.add_argument("name", help="env-var name of the secret")

    del_p = subparsers.add_parser("delete", help="remove a stored secret")
    del_p.add_argument("name", help="env-var name of the secret")

    subparsers.add_parser("list", help="list stored secret names")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not has_keyring():
        print(
            "No OS credential-manager backend is available. "
            "Fall back to environment variables (see .env.example).",
            file=sys.stderr,
        )
        return 1

    if args.command == "set":
        value = args.value
        if value is None:
            value = getpass.getpass(f"{args.name}: ")
        if not value:
            print("empty value ignored", file=sys.stderr)
            return 1
        if set_secret(args.name, value):
            print(f"stored {args.name} in the system credential manager")
            return 0
        print(f"failed to store {args.name}", file=sys.stderr)
        return 1

    if args.command == "get":
        value = get_secret(args.name)
        if value:
            print(value)
            return 0
        print(f"no stored value for {args.name}", file=sys.stderr)
        return 1

    if args.command == "delete":
        delete_secret(args.name)
        print(f"deleted {args.name}")
        return 0

    if args.command == "list":
        print("\n".join(sorted(_KNOWN_SECRETS)))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

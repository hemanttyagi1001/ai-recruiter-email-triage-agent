"""
CLI entrypoint for the kill switch. Thin shim that forwards to
app.kill_switch — this file exists so `python -m app.cli.halt` works
alongside `python -m app.cli.ingest`, `.digest`, etc.
"""

from app.kill_switch import _cli

if __name__ == "__main__":
    import sys
    sys.exit(_cli())

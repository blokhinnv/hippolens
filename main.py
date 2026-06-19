"""Root-level CLI shim; prefer `uv run hippolens` or project scripts."""

from hippolens.cli import main

if __name__ == "__main__":
    main()

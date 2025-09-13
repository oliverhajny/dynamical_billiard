"""
Entry point for running billiard simulations as a module.

Usage examples:
    python -m billiard --shape ellipse --a 5 --ecc 0.6 --animate
    python -m billiard --shape stadium --R 1.0 --L 2.0 --n-starts 7

This forwards to `billiard.initiate.main()` which provides the CLI.
"""

from .initiate import main


if __name__ == "__main__":
    main()


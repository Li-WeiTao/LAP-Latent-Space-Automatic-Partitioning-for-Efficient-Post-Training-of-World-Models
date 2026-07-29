"""Compatibility entry point for the vendored LeWM trainer."""

from backends.lewm.vendor.train import *  # noqa: F401,F403


if __name__ == "__main__":
    run()

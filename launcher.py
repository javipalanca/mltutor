#!/usr/bin/env python3
"""Compatibility entry point for the native portable Qt application."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mltutor.desktop.__main__ import main

    raise SystemExit(main())

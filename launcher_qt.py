#!/usr/bin/env python3
"""Portable MLTutor entry point (no browser, server or installer)."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mltutor.desktop.__main__ import main

    raise SystemExit(main())

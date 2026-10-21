#!/usr/bin/env python3
"""Legacy command name; MLTutor now opens directly in Qt."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mltutor.desktop.__main__ import main

    raise SystemExit(main())

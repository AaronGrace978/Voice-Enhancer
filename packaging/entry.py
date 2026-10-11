"""PyInstaller entry. A double-click starts the booth."""

import multiprocessing

from vocallift.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

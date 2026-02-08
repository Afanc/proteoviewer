import sys, os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

def _init_early_logging(app_name="ProteoViewer", filename="proteoviewer.log"):
    try:
        # Folder of the frozen EXE; fallback to CWD if unfrozen (dev)
        if getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).resolve().parent
        else:
            base_dir = Path.cwd()

        log_path = base_dir / filename

        # Probe write permission; fallback to LOCALAPPDATA if blocked (Program Files)
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8"):
                pass
        except Exception:
            alt = (os.getenv("LOCALAPPDATA")
                   or os.getenv("APPDATA")
                   or str(Path.home()))
            base_dir = Path(alt) / app_name
            base_dir.mkdir(parents=True, exist_ok=True)
            log_path = base_dir / filename

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        # Only add once (PyInstaller can init multiple interpreters internally)
        if not any(isinstance(h, RotatingFileHandler) and getattr(h, "_pf_tag", None) == "early" for h in root.handlers):
            fh = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            fh._pf_tag = "early"
            root.addHandler(fh)

            # Redirect print()/tracebacks very early
            class _StreamToLogger:
                def __init__(self, level): self.level = level
                def write(self, buf):
                    for line in buf.rstrip().splitlines():
                        logging.log(self.level, line)
                def flush(self): pass

            sys.stdout = _StreamToLogger(logging.INFO)
            sys.stderr = _StreamToLogger(logging.ERROR)

            def _excepthook(exc_type, exc, tb):
                logging.exception("Unhandled exception (early)", exc_info=(exc_type, exc, tb))
                try:
                    print(f"A fatal error occurred. See log at: {log_path}")
                except Exception:
                    pass
            sys.excepthook = _excepthook

            logging.getLogger("bokeh").setLevel(logging.INFO)
            logging.getLogger("panel").setLevel(logging.INFO)
            logging.getLogger("tornado").setLevel(logging.INFO)

    except Exception:
        # Last-ditch: never raise from a hook
        pass

_init_early_logging()


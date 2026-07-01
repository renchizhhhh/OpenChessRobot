from pathlib import Path
import sys


def prefer_source_scripts(script_file):
    """Prefer source modules over catkin executable relay scripts."""
    scripts_dir = str(Path(script_file).resolve().parent)
    if scripts_dir in sys.path:
        sys.path.remove(scripts_dir)
    sys.path.insert(0, scripts_dir)

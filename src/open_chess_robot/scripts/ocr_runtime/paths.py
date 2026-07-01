from functools import lru_cache
import os
from pathlib import Path


PACKAGE_NAME = "open_chess_robot"


@lru_cache(maxsize=1)
def package_root():
    override = os.getenv("MOVE_CHESS_PANDA_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    try:
        import rospkg
    except ImportError:
        rospkg = None

    if rospkg is not None:
        try:
            return Path(rospkg.RosPack().get_path(PACKAGE_NAME))
        except rospkg.ResourceNotFound:
            pass

    for source_root in Path(__file__).resolve().parents:
        if (source_root / "package.xml").is_file():
            return source_root
    raise RuntimeError(f"Unable to locate ROS package '{PACKAGE_NAME}'")


def resource_path(*parts):
    return package_root().joinpath(*parts)


def user_data_path(*parts):
    ros_home = Path(os.getenv("ROS_HOME", Path.home() / ".ros")).expanduser()
    path = ros_home.joinpath(PACKAGE_NAME, *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def user_data_dir(*parts):
    path = user_data_path(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

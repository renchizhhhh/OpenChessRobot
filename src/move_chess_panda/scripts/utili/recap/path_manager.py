"""Module for conveniently managing paths through the :class:`URI` class which is fully compatible with :class:`pathlib.Path`.
"""

from pathlib import Path, _PosixFlavour
from typing import Callable
import logging
import abc
import re
import os
import wrapt

logger = logging.getLogger(__name__)


class _URIFlavour(_PosixFlavour):
    has_drv = True
    is_supported = True

    def splitroot(self, part, sep=_PosixFlavour.sep):
        assert sep == self.sep

        match = re.match(rf"(.*):{re.escape(sep)}{{2}}(.*)", part)
        if match:
            drive, path = match.groups()
            drive = drive + "://"
            root = ""
            return drive, root, path
        else:
            self.has_drv = False
            return super().splitroot(part, sep=sep)


class _URIBase(Path):
    _flavour = _URIFlavour()

    @property
    def scheme(self) -> str:
        if not self.drive:
            return ""
        return self.drive[:-len("://")]

    @property
    def path(self) -> str:
        begin = 1 if self.drive or self.root else 0
        return self.root + self._flavour.join(self.parts[begin:])

    def __repr__(self) -> str:
        s = ""
        if self.scheme:
            s += self.scheme + ":" + self._flavour.sep * 2
        s += self.path
        return "{}({!r})".format(self.__class__.__name__, s)


class PathTranslator(abc.ABC):
    """Abstract class representing a path translator that can translate a specific type of URIs to local paths.
    """

    def __call__(self, uri: "URI") -> Path:
        """Translate a URI to a local path.

        Usually, this involves looking at uri.path.

        Args:
            uri (URI): the URI

        Returns:
            Path: the corresponding local path
        """

        raise NotImplementedError


class PathManagerBase:
    """Base class for a path manager.

    This class simultaneously acts as a context manager for the currently active path manager of the :class:`URI` class.
    """

    def __init__(self):
        self._handlers = {}
        self._previous_path_managers = []

    def resolve(self, path: os.PathLike) -> Path:
        """Resolve a path (which might be a :class:`pathlib.Path`) to a local path.

        Args:
            path (os.PathLike): the path

        Returns:
            Path: the corresponding local path
        """

        if not isinstance(path, _URIBase):
            path = _URIBase(path)
        if path.scheme:
            if path.scheme not in self._handlers:
                raise NotImplementedError(
                    f"No handler is registered for scheme {path.scheme}")
            return self._handlers[path.scheme](path)
        else:
            return Path(path.path)

    def register_handler(self, scheme: str) -> Callable[[PathTranslator], PathTranslator]:
        """Decorator to register a path handler for a given URI scheme.

        Args:
            scheme (str): the scheme

        Returns:
            callable: the decorated function
        """

        def decorator(translator: PathTranslator) -> PathTranslator:
            self._handlers[scheme] = translator
            logger.debug(f"Registered path handler for scheme {scheme}")
            return translator
        return decorator

    def __enter__(self):
        self._previous_path_managers.append(PathManager.__wrapped__)
        PathManager.__wrapped__ = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        PathManager.__wrapped__ = self._previous_path_managers.pop()


#: The public path manager instance.
PathManager: PathManagerBase = wrapt.ObjectProxy(PathManagerBase())


class URI(_URIBase):
    """A class representing a recap URI that is lazily evaluated to a local path when it is used.

    It is fully compatible with :class:`pathlib.Path`.
    """

    def __new__(cls, *args):
        return super(URI, cls).__new__(cls, *args)
    
    def __init__(self, *args):
        super().__init__()
        self._local_path = PathManager.resolve(self) 

    def __str__(self) -> str:
        if not hasattr(self, "_local_path"):
            self.__init__()
        return str(self._local_path)

    def is_absolute(self) -> bool:
        return self._local_path.is_absolute()


def register_translator(scheme: str, path: Path):
    """Convenience method to register a path translator that forwards a URI scheme to a local path.

    Args:
        scheme (str): the URI scheme
        path (Path): the local path
    """

    class Translator(PathTranslator):
        def __call__(self, uri: URI) -> Path:
            return path / uri.path
    PathManager.register_handler(scheme)(Translator())
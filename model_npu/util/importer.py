# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import inspect
import pkgutil
import sys
from typing import Callable


def import_packages(package_name: str, blacklist_pkgs: list[str] | None = None) -> list[str]:
    """Import all sub-packages in a package recursively.

    It is easier to use this function to import all sub-packages in a package recursively
    than to manually import each sub-package.

    It replaces the need of the following code snippet on the top of each package's ``__init__.py`` file:

    .. code-block:: python

        import .locomotion.velocity
        import .manipulation.reach
        import .manipulation.lift

    Args:
        package_name: The package name.
        blacklist_pkgs: The list of blacklisted packages to skip. Defaults to None,
            which means no packages are blacklisted.

    Returns:
        A list of class names found in the imported modules, suitable for use in __all__.
    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []
    # Import the package itself
    package = importlib.import_module(package_name)

    # Collect all class names and objects
    class_names = []
    classes = {}

    # Import all Python files and collect classes
    for info in _walk_packages(package.__path__, package.__name__ + ".", blacklist_pkgs=blacklist_pkgs):
        try:
            module = importlib.import_module(info.name)
            # Get all classes defined in this module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Only include classes that are actually defined in this module (not imported)
                if obj.__module__ == info.name and not name.startswith('_'):
                    class_names.append(name)
                    classes[name] = obj
        except Exception:
            # Skip modules that fail to import
            pass

    # Inject classes into the package's namespace
    package_module = sys.modules[package_name]
    for name, obj in classes.items():
        setattr(package_module, name, obj)

    return sorted(set(class_names))


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror: Callable | None = None,
    blacklist_pkgs: list[str] | None = None,
):
    """Yields ModuleInfo for all modules recursively on path, or, if path is None, all accessible modules.

    Note:
        This function is a modified version of the original ``pkgutil.walk_packages`` function. It adds
        the ``blacklist_pkgs`` argument to skip blacklisted packages. Please refer to the original
        ``pkgutil.walk_packages`` function for more details.

    """
    # Default blacklist
    if blacklist_pkgs is None:
        blacklist_pkgs = []

    def seen(p: str, m: dict[str, bool] = {}) -> bool:
        """Check if a package has been seen before."""
        if p in m:
            return True
        m[p] = True
        return False

    for info in pkgutil.iter_modules(path, prefix):
        # check blacklisted
        if any([black_pkg_name in info.name for black_pkg_name in blacklist_pkgs]):
            continue

        # yield the module info
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path: list = getattr(sys.modules[info.name], "__path__", [])

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from _walk_packages(path, info.name + ".", onerror, blacklist_pkgs)

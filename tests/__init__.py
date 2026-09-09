"""Test package.

Made a package deliberately: it lets the modules share helpers through
``from .conftest import make_vitals`` and keeps test module names from colliding with
anything importable, which is the failure mode of a bare ``tests/`` directory.
"""

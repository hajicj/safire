#!/usr/bin/env python
"""
This package contains prepared scenarios for some standard tasks.
A scenario is a script with the following structure:

* build pipeline
* execute pipeline

Instead of defining a format for pipeline settings, we will store the settings
already in Python. Each scenario defines its own settings. (This mechanism
may change yet.) You can define some settings to be controlled by command line,
although we do not recommend doing this too much -- too general scenarios will
lead to command line clutter.

"""
__author__ = 'Jan Hajic jr.'

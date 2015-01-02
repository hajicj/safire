"""
This module contains the base parameter space class.
Marked for deprecation.
"""
import argparse
import collections
import logging

__author__ = "Jan Hajic jr."


class ParameterSpace(object):
    """This class is the base class for Parameter spaces. Parameter spaces
    define the *axes* for experimental parameters. One parameter represents
    one axis of the parameter space.

    Parameter Space objects are used in managing experiments. They are utilized
    in wrappers around experiment scripts to automatically run the experiment
    with various parameter settings.

    Parameter spaces specify two things:

    * What axes are available and what the allowed values are,

    * What naming conventions the experiment uses.

    The basic functionality parameter spaces provide is the :meth:`walk` method:
    an iteration over all parameter configurations along the specified axes.
    """

    def __init__(self):

        # An ArgumentParser of all available axes. Indexed by axis name. Values
        # are iterables of possible parameter values. The default value
        # for each axis is simply the first value.
        self.axes = self.build_parser()

    def walk(self, axes):
        """Generates a cartesian product of parameter configurations along
        the specified axes. The axes that are not given to walk over will
        always have their default value."""

        # ?? Can choices be accessed like this?
        requested_axes = [ self.axes.ax.choices for ax in axes ]
        default_axes = [ self.axes.ax.default for ax in self.axes if ax not in axes ]

    def build_parser(self):
        """Returns a default argparse.ArgumentParser that defines the parameter
        space.
        """
        parser = argparse.ArgumentParser()

        return parser

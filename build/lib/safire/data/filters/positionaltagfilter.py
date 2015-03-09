#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from .basefilter import BaseFilter



logger = logging.getLogger(__name__)


class PositionalTagTokenFilter(BaseFilter):
    """Implements a positional tag-based filter, either as an exclusion
    filter or an inclusion filter (inclusion, by default: only listed values
    are retained). Works by matching a tag string's position against a set of
    permitted/forbidden values.

    Working with the UFAL positional tag set and tagged text in vertical
    format as output by MorphoDiTa, we can set up a filter for autosemantic
    words like this:

    >>> f = PositionalTagTokenFilter(['N','A','D','V'], 0)

    (Defaults enable cooperation with defaults of VTextCorpus seamlesssly.)
    """
    def __init__(self, values, t_position, inclusive=True,
                 colnames=['form','lemma','tag'], tag_colname='tag'):
        """Initilizes the filter.

        :type values: list
        :param values: A list of items that should match what the method
            ``extract_value(fields)`` returns.

        :type t_position: int
        :param t_position: Which position of the tag should be checked for
            presence/absence in given ``values`` list.

        :type inclusive: bool
        :param inclusive: If True (default), will let pass those items
            that have their POS in the ``pos`` list. If False, will filter
            the given parts of speech out.

        :type colnames: list
        :param colnames: A list of column names used together with 
            ``tag_colname`` to determine which column to use as the tag.

        :type tag_colname: str
        :param tag_colname: One of the ``colnames``. The one that contains the
            tag itself. Defaults cooperate seamlessly with VTextCorpus.
        """
        self.values = set(values)
        self.inclusive = inclusive
        self.colnames = colnames

        self.tag_colname = tag_colname
        if self.tag_colname not in colnames:
            raise ValueError('Tag column name %s not in column names %s.' % (tag_colname, colnames))

        self._tag_col_idx = self.colnames.index(self.tag_colname)

        self.t_position = t_position

    def passes(self, fields):
        """Implements the filtering. Returns True on item passing through the
        filter, False on item being filtered out."""
        value = self._extract_tag_position(fields)

        return (value in self.values) == self.inclusive

    def _extract_tag_position(self, fields):
        """Based on the ``init()`` parameters ``colnames``, ``tag_colname``
        and ``t_position``, returns the value at the tag position that
        should be checked.
        """
        return fields[self._tag_col_idx][self.t_position]

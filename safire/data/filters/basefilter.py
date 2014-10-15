#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from gensim.utils import SaveLoad

logger = logging.getLogger(__name__)

class BaseFilter(SaveLoad):

    def __call__(self, fields):
        """Based on the fields, returns True (passes filter)
        or False (stopped by filter).

        :type fields: list
        :param fields: A list of items based on which the filter decides
            whether the item passes or not. Internally calls the ``passes()``
            method, so that it can be integrated without being called
            directly.
        """
        return self.passes(fields)


    def passes(self, fields):
        """Based on the fields, returns True (passes filter)
        or False (stopped by filter). Used by the ``__call__()`` method
        when the filter is used as a callable (e.g. in ``VTextCorpus``).

        :type fields: list
        :param fields: A list of items based on which the filter decides
            whether the item passes or not. Internally calls the ``passes()``
            method, so that it can be integrated without being called
            directly.
        """
        return True

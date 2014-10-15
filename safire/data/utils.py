#!/usr/bin/env python
# -*- coding: utf-8 -*-

def concat_sparse(sparse1, sparse2, dim1, dim2):
    """Concatenates two sparse gensim vectors. The dimension of the first
    will be added to all keys of the second.
    """
    result = sparse1
    concat_sparse2 = [ (s[0] + dim1, s[1]) for s in sparse2 ]
    result.extend(concat_sparse2)

    return result
# coding: utf-8
# pylint: disable= invalid-name,  unused-import
"""For compatibility"""

from __future__ import absolute_import
import sys

PY3 = (sys.version_info[0] == 3)

if PY3:
    import pickle
    from urllib.request import urlretrieve

    def pkl_load(pk_file):
        return pickle.load(pk_file, encoding='latin1')
else:
    import cPickle as pickle
    from urllib import urlretrieve

    def pkl_load(pk_file):
        return pickle.load(pk_file)

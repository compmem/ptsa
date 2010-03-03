.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PTSA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

**ptsa** (pronounced pizza) is a Python_ module for performing time series
analysis. Although it is specifically designed with neural data in
mind (EEG, MEG, fMRI, etc. ...), the code should be applicable to almost
any type of time series.

.. _Python: http://www.python.org

**ptsa** stands for **P**\ ython\ **T**\ ime **S**\ eries **A**\ nalysis.


.. News
.. ====

.. None yet...

Documentation
=============

* :ref:`Main data structures in ptsa <dimarray>`: The documentation
  for AttrArray, Dim, and DimArray -- powerful data structures based
  on NumPy ndarrays.

* :ref:`Analysis of time series data <timeseries>`: The documentation
  of TimeSeries, a subclass of DimArray for storing and analyzing data
  with a time dimension.

.. * `Installation Instructions`: to come...
* :ref:`Developer Guidelines <devguide>` (information for people
  contributing code)

.. * `API Reference`_ (comprehensive and up-to-date information about the details
..   of the implementation)
* :ref:`genindex` (access by keywords)
* :ref:`search` (online and offline full-text search)

.. _API Reference: api/index.html

.. * `PTSA Manual (PDF)`_ (all documentation except for developer guidelines
..   and API reference)
.. * `Developer Guidelines (PDF)`_

.. _Main data structures in PTSA: PTSA-Manual.pdf
.. _PTSA Manual (PDF): PTSA-Manual.pdf
.. _Developer Guidelines (PDF): PTSA-DevGuide.pdf


License
=======

PTSA is free-software and covered by the `GPLv3 License`_.
This applies to all source code, documentation, examples and snippets inside
the source distribution (including this website). 

.. Please see the
.. :ref:`appendix of the manual <license>` for the copyright statement and the
.. full text of the license.

.. _GPLv3 License: http://www.gnu.org/licenses/gpl-3.0.html
.. .. _appendix of the manual: manual.html#license



.. Download
.. ========

.. Binary packages
.. ---------------



.. Source code
.. -----------

.. Source code tarballs of PTSA releases are available from the `download
.. area`_. Alternatively, one can also download a tarball of the latest
.. development snapshot_ (i.e. the current state of the *master* branch of the
.. PTSA source code repository).

.. To get access to both the full PTSA history and the latest
.. development code, the PTSA Git_ repository is publicly available. To view the
.. repository, please point your webbrowser to gitweb:
.. http://tbd

.. To clone (aka checkout) the PTSA repository simply do:

.. ::

..   git clone git://tbd

.. After a short while you will have a `ptsa` directory below your current
.. working directory, that contains the PTSA repository.

.. More detailed instructions on :ref:`installation requirements <requirements>`
.. and on how to :ref:`build PTSA from source <buildfromsource>` are provided
.. in the manual.


.. .. _download area: http://tbd
.. .. _Git: http://git.or.cz/
.. .. _snapshot:  http://tbd


.. Support
.. =======

.. If you have problems installing the software or questions about usage,
.. documentation or something else related to PTSA, you can post to the PTSA
.. mailing list:

.. :Mailing list: tbd@tbd [subscription_,
..                archive_]

.. All users should subscribe to the mailing list. PTSA is still a young project
.. that is under heavy development. Significant modifications (hopefully
.. improvements) are very likely to happen frequently. The mailing list is the
.. preferred way to announce such changes. The mailing list archive can also be
.. searched using the *mailing list archive search* located in the sidebar of the
.. PTSA home page.

.. .. _subscription: http://tbd
.. .. _archive: http://tbd



.. Publications
.. ============

.. .. .. include:: publications.txt


.. Authors & Contributors
.. ======================

.. .. .. include:: authors.txt


.. Similar or Related Projects
.. ===========================

.. -*- mode: rst; fill-column: 79 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _devguide:

*************************
PTSA Developer Guidelines
*************************


Documentation
=============

Documentation of the code and supplementary material (such as this file)
should be done in reST_ (reStructuredText) light markup language.  See `Demo
<http://docutils.sourceforge.net/docs/user/rst/cheatsheet.txt>`__ or a
`Cheatsheet <http://docutils.sourceforge.net/docs/user/rst/demo.txt>`__ for a
quick demo.


Code Documentation
------------------
PTSA follows the `NumPy/SciPy documentation guidelines`_.
Example docstrings are discussed in the guidlines and available as separate
files from the `SciPy`_ website:

* `example.py`_
* `EXAMPLE_DOCSTRING.txt`_

.. _NumPy/SciPy documentation guidelines: http://projects.scipy.org/scipy/numpy/wiki/CodingStyleGuidelines

.. _SciPy: http://scipy.org

.. _example.py: http://projects.scipy.org/scipy/numpy/browser/trunk/doc/example.py

.. _EXAMPLE_DOCSTRING.txt: http://projects.scipy.org/scipy/numpy/browser/trunk/doc/EXAMPLE_DOCSTRING.txt


Code Formatting
===============

pylint
   Code should be conformant with Pylint_ driven by config located at
   `doc/misc/pylintrc <misc/pylintrc>`__.  It assumes camelback notation
   (classes start with capitals, functions with lowercase) and indentation
   using 4 spaces (i.e. no tabs) Variables are low-case and can have up to 2
   _s. To engage, use 1 of 3 methods:

   - place it in *~/.pylintrc* for user-wide installation
   - use within a call to pylint::

       pylint --rcfile=$PWD/doc/misc/pylintrc

   - export environment variable from mvpa sources top directory::

       export   PYLINTRC=$PWD/doc/misc/pylintrc

module docstring
   Each module should start with a docstring describing the module.

notes
   Use following keywords will be caught by pylint to provide a
   summary of what yet to be done in the given file

   FIXME
     something which needs fixing (sooner than later)
   TODO
     future plan (i.e. later than sooner)
   XXX
     some concern/question
   YYY
     comment/answer to above mentioned XXX concern
   WiP
     Work in Progress: API and functionality might rapidly change



Coding Conventions
==================

__repr__
  most of the classes should provide meaningful and concise summary
  over their identity (name + parameters + some summary over results
  if any)


Tests
=====

* Every more or less "interesting" bugfix should be accompanied by a
  unittest which might help to prevent it in the future refactoring
* Every new feature should have a unittest



Git Repository
==============

Layout
------

The repository is structured by a number of branches. Each developer should
prefix his/her branches with a unique string plus '/' (maybe initials or
similar). Currently there are:

  :per: Per B. Sederberg
  :ctw: Christoph T. Weidemann

The main release branch is called *master*. This is a merge-only branch.
Features finished or updated by some developer are merged from the
corresponding branch into *master*. At a certain point the current state of
*master* is tagged -- a release is done.

Only usable feature should end-up in *master*. Ideally *master* should be
releasable at all times. Something must not be merged into master if *any*
unit test fails.

Additionally, there are packaging branches. They are labeled after the package
target (e.g. *debian* for a Debian package). Releases are merged into the
packaging branches, packaging get updated if necessary and the branch gets
tagged when a package version is released. Maintenance (as well as backport)
releases should be done under *maint/codename.flavor* (e.g. *maint/lenny*,
*maint/lenny.security*, *maint/sarge.bpo*).


Commits
-------

Please prefix all commit summaries with one (or more) of the following labels.
This should help others to easily classify the commits into meaningful
categories:

  * *BF* : bug fix
  * *RF* : refactoring
  * *NF* : new feature
  * *OPT* : optimization
  * *BK* : breaks something and/or tests fail
  * *PL* : making pylint happier
  * *DOC*: for all kinds of documentation related commits

.. _reST: http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html
.. _EmacsreST: http://docutils.sourceforge.net/docs/user/emacs.html
.. _Pylint: http://packages.debian.org/unstable/python/pylint


Merges
------

For easy tracking of what changes were absorbed during merge, we
advice to enable merge summary within git:

  git-config merge.summary true


Changelog
=========

The PTSA changelog is located in the toplevel directory of the source tree
in the `Changelog` file. The content of this file should be formated as
restructured text to make it easy to put it into manual appendix and on the
website.

This changelog should neither replicate the VCS commit log nor the
distribution packaging changelogs (e.g. debian/changelog). It should be
focused on the user perspective and is intended to list rather macroscopic
and/or important changes to the module, like feature additions or bugfixes in
the algorithms with implications to the performance or validity of results.

It may list references to 3rd party bugtrackers, in case the reported bugs
match the criteria listed above.

Changelog entries should be tagged with the name of the developer(s) (mainly)
involved in the modification -- initials are sufficient for people
contributing regularly.

Changelog entries should be added whenever something is ready to be merged
into the master branch, not necessarily with a release already approaching.

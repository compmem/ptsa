.. -*- mode: rst -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PTSA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _dimarray:

.. index:: AttrArray, Attribute Array, Dim, Dimension, DimArray, Dimensioned Array

****************************
Main data structures in PTSA
****************************

.. index:: AttrArray, Attribute Array, NumPy, ndarray

.. _AttrArray:

Attribute Array (AttrArray)
===========================

An Attribute Array is simply a `NumPy ndarray`_ which allows the specification
of custom attributes. These attributes can be specified as keyword arguments
or set and changed on the fly as shown in the example below.

  >>> import numpy as np
  >>> import dimarray as da
  >>> data = da.AttrArray(np.random.rand(5), hello='world')
  >>> print data.hello
  world
  >>> data.hello = 'good bye'
  >>> print data.hello
  good bye
  >>> data.version = 1.0
  >>> print data.version
  1.0

These custom attributes stick around when you copy or manipulate the
data in a :class:`AttrArray`:

  >>> data2 = data.mean()
  >>> data2.hello
  good bye

:class:`AttrArray` instances are initialized just like
:class:`ndarray` instances but they accept arbitrary keyword arguments
that can be used to specify custom attributes during
initialization. Every :class:`AttrArray` has a protected (read-only)
:attr:`_required_attrs` attribute, which is :obj:`None` when no
attributes are required (as is the case for instances of
:class:`AttrArray`) or a :class:`dictionary` that specifies required
attributes (for child classes of :class:`AttrArray`, such as
:class:`Dim` and :class:`DimArray`).

:class:`AttrArray` instances can be used to store information about the data
they contain. The main purpose of :class:`AttrArray` in :mod:`PTSA` is to
serve as a base class for more specialized data structures that require
certain attributes to contain specific information. These data structures are
described below.

.. _NumPy ndarray: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

.. _Dim:

Dimension (Dim)
===============

:class:`Dim` is a child class of :class:`AttrArray` with the
constraints that each instance be 1-dimensional and have a
:attr:`name` attribute. If multi dimensional input is specified during
initialization, an attempt is made to convert it to one dimension by
collapsing over dimensions that only have one level (if that fails an
error is raised):

  >>> import numpy as np
  >>> import dimarray as da
  >>> test = da.Dim([[1,2,3]], name='dimension 1')
  >>> print test
  [1 2 3]

:class:`Dim` instances are part of :class:`DimArray` instances and
within :class:`DimArray` instances the following additional
constraints are enforced:
 * Values within a :class:`Dim` instance must be unique.
 * No two :class:`Dim` instances within the same `DimArray` instance
   may have the same :attr:`name` attribute.


Dimensioned Array (DimArray)
============================

A dimensioned array.

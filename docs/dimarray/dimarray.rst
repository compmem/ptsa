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
  >>> import dimarray
  >>> data = dimarray.AttrArray(np.random.rand(5), hello='world')
  >>> print data.hello
  world
  >>> data.hello = 'good bye'
  >>> print data.hello
  good bye
  >>> data.version = 1.0
  >>> print data.version
  1.0

:class:`AttrArray` instances are initialized just like :class:`ndarray`
instances but accept arbitrary keyword arguments that can be used to specify
custom attributes during initalization. Every :class:`AttrArray` has a
protected (read-only) :attr:`_required_attrs` attribute which is :obj:`None`
when no attributes are required (as is the case for instances of
:class:`AttrArray`) or a :class:`dictionary` that specifies required
attributes (for child classes of :class:`AttrArray`, such as :class:`Dim` and
:class:`DimArray`).

:class:`AttrArray` instances can be used to store information about the data
they contain. The main purpose of :class:`AttrArray` in :mod:`PTSA` is to
serve as a base class for more specialized data structures that require
certain attributes to contain specific information. These data structures are
described below.

.. _NumPy ndarray: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

.. _Dim:

Dimension (Dim)
===============

A dimension.

Dimensioned Array (DimArray)
============================

A dimensioned array.
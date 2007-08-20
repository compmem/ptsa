


class InfoArray(N.ndarray):
    def __new__(subtype, data, info=None, dtype=None, copy=True):
        # When data is an InfoArray
        if isinstance(data, InfoArray):
            if not copy and dtype==data.dtype:
                return data.view(subtype)
            else:
                return data.astype(dtype).view(subtype)
        subtype._info = info
        subtype.info = subtype._info
        return N.array(data).view(subtype)

    def __array_finalize__(self,obj):
        if hasattr(obj, "info"):
            # The object already has an info tag: just use it
            self.info = obj.info
        else:
            # The object has no info tag: use the default
            self.info = self._info

    def __repr__(self):
        desc="""\
array(data=
  %(data)s,
      tag=%(tag)s)"""
        return desc % {'data': str(self), 'tag':self.info }


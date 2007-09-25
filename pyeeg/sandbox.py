
# BaseDict code from: 
#   http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/473790
# import cPickle
# import moved to top of file

class BaseDict(dict):
    '''
        A dict allows inputting data as adict.xxx as well as adict['xxx']

        In python obj:
        
            obj.var=x  ---> 'var' be in obj.__dict__ (this is python default)

        In python dict:
        
            dict['var']=x ---> 'var' be in dict.items(this is dict behavior)

        In BaseDict:  

            let bd=BaseDict()
            
            Both bd.var and bd['var'] will save x to bd.items 
            and bd.setDict('var', x) will save x to bd.__dict__ 

            This allows an easier access of the variables.        
  
    '''
    def __init__(self, data=None):
        if data:  dict.__init__(self, data)
        else:     dict.__init__(self)
        dic = self.__dict__
        dic['__ver__']   ='20041208_1'
        dic['__author__']='Runsun Pan'
    
    def __setattr__(self, name, val):
        if name in self.__dict__:  self.__dict__[name]= val        
        else:   self[name] = val
        
    def __getattr__(self, name):
        if name in self.__dict__:  return self.__dict__[name]        
        else:  return self[name] 
           
    def setDict(self, name, val): 
        '''
            setDict(name, val): Assign *val* to the key *name* of __dict__.
         
            :Usage:
            
            >>> bd = BaseDict()
            >>> bd.getDict()['height']   
            Traceback (most recent call last):
            ...
            KeyError: 'height'
            >>> bd.setDict('height', 160)  # setDict 
            {}
            >>> bd.getDict()['height']
            160

            '''
        self.__dict__[name] = val
        return self 

    def getDict(self): 
        ''' 
            Return the internal __dict__.
            
            :Usage:
            
            >>> bd = BaseDict()
            >>> bd.getDict()['height']
            Traceback (most recent call last):
            ...
            KeyError: 'height'
            >>> bd.setDict('height', 160)
            {}
            >>> bd.getDict()['height']
            160
            '''
        return self.__dict__
        
    def setItem(self, name, val): 
        ''' 
            Set the value of dict key *name* to *val*. Note this dict 
            is not the __dict__.

            :Usage:
            
            >>> bd = BaseDict()
            >>> bd
            {}
            >>> bd.setItem('sex', 'male')
            {'sex': 'male'}
            >>> bd['sex'] = 'female'
            >>> bd
            {'sex': 'female'}
            '''
        self[name] = val
        return self
    
    def __getstate__(self): 
        ''' Needed for cPickle in .copy() '''
        return self.__dict__.copy() 

    def __setstate__(self,dict): 
        ''' Needed for cPickle in .copy() '''
        self.__dict__.update(dict)   

    def copy(self):   
        ''' 
            Return a copy. 
            
            :Usage:
            
            >>> bd = BaseDict()
            >>> bd['name']=[1,2,3]
            >>> bd
            {'name': [1, 2, 3]}
            >>> bd2 = bd.copy()
            >>> bd2
            {'name': [1, 2, 3]}
            >>> bd == bd2
            True
            >>> bd is bd2
            False
            >>> bd['name']==bd2['name']
            True
            >>> bd['name'] is bd2['name']
            False
            >>> bd2['name'][0]='aa'
            >>> bd2['height']=60
            >>> bd
            {'name': [1, 2, 3]}
            >>> bd2
            {'name': ['aa', 2, 3], 'height': 60}
                
            '''
        return cPickle.loads(cPickle.dumps(self))



class DataDict(BaseDict):
    """ Dictionary where you can access the values as attributes, but with
    added features for manipulating the data inside.  """
    def removeBuffer(self,fields,axis=-1):
	"""Use the information contained in the data dictionary to remove the
	buffer from the specified fields and reset the time range.  If
	bufLen is 0, no action is performed."""
	# see if remove the anything
	if self.bufLen>0:
	    # make sure it's a list
	    fields = N.asarray(fields)
	    if len(fields.shape)==0:
		fields = [fields]
	    for field in fields:
		# remove the buffer
		self[field] = self[field].take(range(self.bufLen,
						     self[field].shape[axis]-self.bufLen),
					       axis)
	    # set the time range with no buffer
	    self.time = N.linspace(self.OffsetMS,
				   self.OffsetMS+self.DurationMS,
				   self[fields[0]].shape[axis])
	    # reset buffer to indicate it was removed
	    self.bufLen = 0



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


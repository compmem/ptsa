
import numpy as np
#from numpy.testing import NumpyTest, NumpyTestCase

from ptsa import Events,TsEvents,DataWrapper

class test_Events:
    def setUp(self):
        self.test1xyz = np.random.random((2,))
        self.test1yz = self.test1xyz.copy()
        self.test1xz = self.test1xyz.copy()
        self.test1xy = self.test1xyz.copy()
        self.test1x = self.test1xyz.copy()
        self.test1y = self.test1xyz.copy()
        self.test1z = self.test1xyz.copy()
        self.test1xyz.dtype = [('x', int), ('y', float), ('z', int)]
        self.test1yz.dtype = [('y', float), ('z', int)]
        self.test1xz.dtype = [('x', int), ('z', int)]
        self.test1xy.dtype = [('x', int), ('y', float)]
        self.test1x.dtype = [('x', int)]
        self.test1y.dtype = [('y', float)]
        self.test1z.dtype = [('z', int)]

        self.test2sox = np.random.random((3,4))
        self.test2so = self.test2sox.copy()
        self.test2sx = self.test2sox.copy()
        self.test2sox1 = self.test2sox.copy()
        self.test2sox2 = self.test2sox.copy()
        self.test2sox3 = self.test2sox.copy()
        self.test2soxy = self.test2sox.copy()
        self.test2soxyz = self.test2sox.copy()
        self.test2soy = self.test2sox.copy()
        self.test2soz = self.test2sox.copy()
        self.test2soyz = self.test2sox.copy()
        self.test2soxz = self.test2sox.copy()
        
        self.test2sox.dtype = [('tssrc', DataWrapper),
                               ('offset', int), ('x', int)]
        self.test2so.dtype = [('tssrc', DataWrapper),
                              ('offset', int)]
        self.test2sx.dtype = [('tssrc', DataWrapper), ('x', int)]

        self.test2sox1.dtype = [('tssrc', int),
                                ('offset', int), ('x', DataWrapper)]
        self.test2sox2.dtype = [('tssrc', DataWrapper),
                                ('offset', DataWrapper), ('x', int)]
        self.test2sox3.dtype = [('tssrc', int),
                                ('offset', int), ('x', DataWrapper)]
        
        self.test2soxyz.dtype = [('tssrc', DataWrapper), ('offset', int),
                                 ('x', int), ('y', float), ('z', str)]
        self.test2soxy.dtype = [('tssrc', DataWrapper), ('offset', int),
                                ('x', int), ('y', float)]
        self.test2soxz.dtype = [('tssrc', DataWrapper), ('offset', int),
                                ('x', int), ('z', str)]
        self.test2soyz.dtype = [('tssrc', DataWrapper), ('offset', int),
                                ('y', float), ('z', str)]
        self.test2soy.dtype = [('tssrc', DataWrapper), ('offset', int),
                               ('y', float)]
        self.test2soz.dtype = [('tssrc', DataWrapper), ('offset', int),
                               ('z', str)]

        #self.test1 = test.view(Events)
        # set up a fake dataset
        #self.tst = Events((2,),dtype=[('x', int), ('y', float), ('z', int)])
    def test_init(self):
        pass

    def test_getitem(self):
        pass

    def test_setitem(self):
        pass

    def test_select(self):
        pass

    def test_extend(self):
        pass

    def test_remove_fields(self):
        pass

    def test_add_fields(self):
        pass

    

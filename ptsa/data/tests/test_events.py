#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


import numpy as np
from numpy.testing import TestCase, assert_array_equal,\
     assert_array_almost_equal

from ptsa.data import ArrayWrapper,BaseWrapper
from ptsa.data.events import Events



class Setup():
    def __init__(self):
        self.test1xyz = np.array([(1.0, 2, 'bla1'), (3.0, 4, 'bla2')],
                                 dtype=[('x', float), ('y', int),
                                        ('z', '|S4')])
        self.test1yz = np.array([(2, 'bla1'), (4, 'bla2')],
                                dtype=[('y', int), ('z', '|S4')])
        self.test1xz = np.array([(1.0, 'bla1'), (3.0, 'bla2')],
                                dtype=[('x', float), ('z', '|S4')])
        self.test1xy = np.array([(1.0, 2), (3.0, 4)],
                                dtype=[('x', float), ('y', int)])
        self.test1x = np.array([(1.0,), (3.0,)],
                               dtype=[('x', float)])
        self.test1y = np.array([(2,), (4,)],
                               dtype=[('y', int)])
        self.test1z = np.array([('bla1',), ('bla2',)],
                               dtype=[('z', '|S4')])

        dw = BaseWrapper()

        self.test2sox = np.array([[(dw, 2, 42.),
                                   (dw, 4, 33.)],
                                  [(dw, 5, 22.),
                                   (dw, 6, 11.)]],
                                 dtype=[('esrc', BaseWrapper),
                                        ('eoffset', int),
                                        ('x', float)])
        self.test2so = np.array([[(dw, 2),
                                  (dw, 4)],
                                 [(dw, 5),
                                  (dw, 6)]],
                                dtype=[('esrc', BaseWrapper),
                                       ('eoffset', int)])
        self.test2sx = np.array([[(dw, 42.),
                                  (dw, 33.)],
                                 [(dw, 22.),
                                  (dw, 11.)]],
                                dtype=[('esrc', BaseWrapper),
                                       ('x', float)])
        self.test2sox1 = np.array([[(1.5, 2, dw),
                                    (1.5, 4, dw)],
                                   [(1.5, 5, dw),
                                    (1.5, 6, dw)]],
                                  dtype=[('esrc', float),
                                         ('eoffset', int),
                                         ('x', BaseWrapper)])
        self.test2sox2 = np.array([[(dw, dw, 42),
                                    (dw, dw, 33)],
                                   [(dw, dw, 22),
                                    (dw, dw, 11)]],
                                  dtype=[('esrc', BaseWrapper),
                                         ('eoffset', BaseWrapper),
                                         ('x', int)])
        self.test2sox3 = np.array([[(3, 2, dw),
                                    (3, 4, dw)],
                                   [(3, 5, dw),
                                    (3, 6, dw)]],
                                  dtype=[('esrc', int),
                                         ('eoffset', int),
                                         ('x', BaseWrapper)])
        self.test2soxy = np.array([[(dw, 2, 42., 1),
                                    (dw, 4, 33., 2)],
                                   [(dw, 5, 22., 3),
                                    (dw, 6, 11., 4)]],
                                  dtype=[('esrc', BaseWrapper),
                                         ('eoffset', int),
                                         ('x', float),('y',int)])

        self.test2soxyz = np.array([[(dw, 2, 42., 1, 'z1'),
                                     (dw, 4, 33., 2, 'z2')],
                                    [(dw, 5, 22., 3, 'z3'),
                                     (dw, 6, 11., 4, 'z4')]],
                                   dtype=[('esrc', BaseWrapper),
                                          ('eoffset', int),
                                          ('x', float),('y',int),('z','|S2')])
        self.test2soy = np.array([[(dw, 2, 1),
                                     (dw, 4, 2)],
                                    [(dw, 5, 3),
                                     (dw, 6, 4)]],
                                 dtype=[('esrc', BaseWrapper),
                                        ('eoffset', int),
                                        ('y',int)])

        self.test2soz = np.array([[(dw, 2, 'z1'),
                                   (dw, 4, 'z2')],
                                  [(dw, 5, 'z3'),
                                   (dw, 6, 'z4')]],
                                 dtype=[('esrc', BaseWrapper),
                                        ('eoffset', int),
                                        ('z','|S2')])

        self.test2soyz = np.array([[(dw, 2, 1, 'z1'),
                                    (dw, 4, 2, 'z2')],
                                   [(dw, 5, 3, 'z3'),
                                    (dw, 6, 4, 'z4')]],
                                  dtype=[('esrc', BaseWrapper),
                                         ('eoffset', int),
                                         ('y', int),('z', '|S2')])

        self.test2soxz = np.array([[(dw, 2, 42., 'z1'),
                                    (dw, 4, 33., 'z2')],
                                   [(dw, 5, 22., 'z3'),
                                    (dw, 6, 11., 'z4')]],
                                  dtype=[('esrc', BaseWrapper),
                                         ('eoffset', int),
                                         ('x', float),('z', '|S2')])


class test_Events(TestCase):
    def setUp(self):
        self.dat = np.random.rand(10,1000)
        self.aw = ArrayWrapper(self.dat,200)
        self.eoffsets = [80,140,270]

    def test_new(self):
        tst = Setup()
        test = tst.test1xyz.view(Events)
        test = tst.test1xy.view(Events)
        test = tst.test1xz.view(Events)
        test = tst.test1yz.view(Events)
        test = tst.test1x.view(Events)
        test = tst.test1y.view(Events)
        test = tst.test1z.view(Events)

        test = tst.test2sox.view(Events)
        test = tst.test2so.view(Events)
        test = tst.test2sx.view(Events)
        test = tst.test2sox1.view(Events)
        test = tst.test2sox2.view(Events)
        test = tst.test2sox3.view(Events)
        test = tst.test2soxy.view(Events)
        test = tst.test2soxyz.view(Events)
        test = tst.test2soxz.view(Events)
        test = tst.test2soyz.view(Events)
        test = tst.test2soy.view(Events)
        test = tst.test2soz.view(Events)

    def test_remove_fields(self):
        tst = Setup()
        test_a = tst.test1xyz.view(Events).remove_fields('z')
        test_b = tst.test1xy.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('y')
        test_b = tst.test1xz.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('x')
        test_b = tst.test1yz.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields(
            'y').remove_fields('z')
        test_b = tst.test1x.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('y','z')
        test_b = tst.test1x.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('z','y')
        test_b = tst.test1x.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields(
            'z').remove_fields('y')
        test_b = tst.test1x.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields(
            'y').remove_fields('x')
        test_b = tst.test1z.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('y','x')
        test_b = tst.test1z.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('x','y')
        test_b = tst.test1z.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields(
            'x').remove_fields('y')
        test_b = tst.test1z.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields(
            'x').remove_fields('z')
        test_b = tst.test1y.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('x','z')
        test_b = tst.test1y.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields('z','x')
        test_b = tst.test1y.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events).remove_fields(
            'z').remove_fields('x')
        test_b = tst.test1y.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xy.view(Events).remove_fields('y')
        test_b = tst.test1x.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xy.view(Events).remove_fields('x')
        test_b = tst.test1y.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xz.view(Events).remove_fields('z')
        test_b = tst.test1x.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xz.view(Events).remove_fields('x')
        test_b = tst.test1z.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1yz.view(Events).remove_fields('z')
        test_b = tst.test1y.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1yz.view(Events).remove_fields('y')
        test_b = tst.test1z.view(Events)
        assert_array_equal(test_a,test_b)

        test_a = tst.test2soxyz.view(Events).remove_fields('z')
        test_b = tst.test2soxy.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('y')
        test_b = tst.test2soxz.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('x')
        test_b = tst.test2soyz.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('y','z')
        test_b = tst.test2sox.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('z','y')
        test_b = tst.test2sox.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'z').remove_fields('y')
        test_b = tst.test2sox.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'y').remove_fields('z')
        test_b = tst.test2sox.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('x','y','z')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('x','z','y')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('y','x','z')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('y','z','x')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('z','y','x')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields('z','x','y')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'x').remove_fields('y').remove_fields('z')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'x').remove_fields('z').remove_fields('y')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'y').remove_fields('x').remove_fields('z')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'y').remove_fields('z').remove_fields('x')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'z').remove_fields('x').remove_fields('y')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events).remove_fields(
            'z').remove_fields('y').remove_fields('x')
        test_b = tst.test2so.view(Events)
        assert_array_equal(test_a,test_b)

    def test_add_fields(self):
        tst = Setup()
        x = np.array([1.0,3.0])
        y = np.array([2,4])
        z = np.array(['bla1','bla2'])
        test_a = tst.test1xyz.view(Events)
        self.assertRaises(ValueError,tst.test1xyz.view(Events).add_fields,x=x)
        self.assertRaises(ValueError,tst.test1xyz.view(Events).add_fields,y=int)
        test_b = tst.test1xy.view(Events).add_fields(z=z)
        assert_array_equal(test_a,test_b)
        test_a = test_a.add_fields(a=int)
        test_b = test_b.add_fields(a=int)
        self.assertTrue(test_a.shape==test_b.shape)
        assert_array_equal(np.sort(test_a.dtype.names),
                           np.sort(test_b.dtype.names))
        for f,v in test_a.dtype.fields.iteritems():
            if f!='a':
                assert_array_equal(test_a[f],test_b[f])
            self.assertTrue(test_a.dtype[f]==test_b.dtype[f])
        test_a = tst.test1xyz.view(Events)
        test_b = tst.test1x.view(Events).add_fields(y=y).add_fields(z=z)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xyz.view(Events)
        test_b = tst.test1x.view(Events).add_fields(y=y,z=z)
        assert_array_equal(np.sort(test_a.dtype.names),
                           np.sort(test_b.dtype.names))
        for f,v in test_a.dtype.fields.iteritems():
            assert_array_equal(test_a[f],test_b[f])
            self.assertTrue(test_a.dtype[f]==test_b.dtype[f])
        for field in test_a.dtype.names:
            assert_array_equal(test_a[field],test_b[field])
        test_a = tst.test1xy.view(Events)
        test_b = tst.test1x.view(Events).add_fields(y=y)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1xz.view(Events)
        test_b = tst.test1x.view(Events).add_fields(z=z)
        assert_array_equal(test_a,test_b)
        test_a = tst.test1yz.view(Events)
        test_b = tst.test1y.view(Events).add_fields(z=z)
        assert_array_equal(test_a,test_b)

        x = np.array([[42.,33.],[22.,11.]])
        y = np.array([[1,2],[3,4]])
        z = np.array([['z1','z2'],['z3','z4']])
        test_a = tst.test2soxyz.view(Events)
        test_b = tst.test2soxy.view(Events).add_fields(z=z)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events)
        test_b = tst.test2sox.view(Events).add_fields(y=y,z=z)
        assert_array_equal(np.sort(test_a.dtype.names),
                           np.sort(test_b.dtype.names))
        for f,v in test_a.dtype.fields.iteritems():
            assert_array_equal(test_a[f],test_b[f])
            self.assertTrue(test_a.dtype[f]==test_b.dtype[f])
        for field in test_a.dtype.names:
            assert_array_equal(test_a[field],test_b[field])
        test_a = tst.test2soxyz.view(Events)
        test_b = tst.test2sox.view(Events).add_fields(y=y).add_fields(z=z)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events)
        test_b = tst.test2so.view(Events).add_fields(x=x).add_fields(
            y=y).add_fields(z=z)
        assert_array_equal(test_a,test_b)
        test_a = tst.test2soxyz.view(Events)
        test_b = tst.test2so.view(Events).add_fields(x=x,y=y,z=z)
        assert_array_equal(np.sort(test_a.dtype.names),
                           np.sort(test_b.dtype.names))
        for f,v in test_a.dtype.fields.iteritems():
            assert_array_equal(test_a[f],test_b[f])
            self.assertTrue(test_a.dtype[f]==test_b.dtype[f])

        
    def test_get_data(self):
        # get data directly from the wrapper
        ed = self.aw.get_event_data(3,self.eoffsets,.5,-.1,.25)

        # create same array by hand:
        ed2 = np.array([self.dat[3,60:161],self.dat[3,120:221],
                        self.dat[3,250:351]])
        assert_array_equal(ed,ed2)
        # get data from a events
        # make some events
        events = np.rec.fromarrays(([self.aw]*len(self.eoffsets),self.eoffsets),
                                   names='esrc,eoffset').view(Events)
        ed3 = events.get_data(3,.5,-.1,.25)

        assert_array_almost_equal(ed[:],ed3[:],decimal=6)

        # get data directly from the wrapper
        ed = self.aw.get_event_data(3,self.eoffsets,.5,.1,.25)

        # create same array by hand:
        ed2 = np.array([self.dat[3,100:201],self.dat[3,160:261],
                        self.dat[3,290:391]])
        assert_array_equal(ed,ed2)
        # get data from a events
        # make some events
        events = np.rec.fromarrays(([self.aw]*len(self.eoffsets),self.eoffsets),
                                   names='esrc,eoffset').view(Events)
        ed3 = events.get_data(3,.5,.1,.25)

        assert_array_almost_equal(ed[:],ed3[:],decimal=6)



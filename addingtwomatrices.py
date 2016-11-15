# coding: utf-8
import numpy
import theano.tensor as K
from theano import function
x = K.dmatrix('x')
y = K.dmatrix('y')
z = x + y
f = function([x,y],z)
f([[1,2],[3,4]],[[10,20],[30,40]])

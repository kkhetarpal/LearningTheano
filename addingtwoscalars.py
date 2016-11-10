#Adding two scalars
import numpy
import theano.tensor as K
from theano import function
a = K.dscalar('a')
b = K.dscalar('b')
c = a+b
f = function([a, b], c])
f = function([a, b], c)
f(30,10)
f(-1,4)
numpy.allclose(f(10,10),20)
numpy.allclose(f(10,10),40)
type(a)
a.type
K.dscalar
from theano import pp
print(pp(c))
numpy.allclose(c.eval({a:1,b:2}),3)
numpy.allclose(c.eval({a:1,b:30}),3)
numpy.allclose(c.eval({a:1,b:30}),31)

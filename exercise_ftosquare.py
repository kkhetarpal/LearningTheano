# coding: utf-8
import theano
var_a = theano.tensor.vector()
var_b = theano.tensor.vector()
var_out = var_a ** 2 + var_b ** 2 + 2*var_a*var_b
f_square = theano.function([var_a,var_b],var_out)
print(f_square([1,2],[4,5]))

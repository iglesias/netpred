linal
=====

Structured SVMs applied to graph learning (or CRFs in particular). I am using
this code to carry out some experiments during my final degree project. A lot
of inspiration has been drawn from [pystruct](http://github.com/amueller/pystruct)
by Andreas Mueller. The code makes use of [Shogun](http://shogun-toolbox.org)'s
structured learning framework [+ info](http://iglesiashogun.wordpress.com/).

Requirements
============

You definetely need Shogun and SWIG. Also, you need to compile Shogun with
directors enabled and at least target python's modular interface. In addition,
with the current state you need to compile Shogun with Mosek support. This
dependency should be easy to remove though by using any of the bundle methods
for SSVMs present in Shogun.

Apart from Shogun dependencies, you need cvxopt for the linear programming
relaxation used to solve the argmax.

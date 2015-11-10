linal
=====

Structured SVMs applied to label sequence learning (the so-called HM-SVM) and
graph learning (or grid CRFs in particular). I am using
this code to carry out some experiments during my final degree project. A lot
of inspiration has been drawn from [pystruct](http://github.com/amueller/pystruct)
by Andreas Mueller for the graph learning part. The HM-SVM implementation in
Shogun I am using here is based on the Matlab code by Gunnar Raetsch and Georg
Zeller [available at mloss](http://mloss.org/software/tags/hmsvm/). The code
makes use of [Shogun](http://shogun-toolbox.org)'s structured learning framework
([+ info](http://iglesiashogun.wordpress.com/)).

Requirements
============

You definetely need Shogun and Swig. Also, you need to compile Shogun with
directors enabled and at least target Python's modular interface. In addition,
with the current state you need to compile Shogun with Mosek support. This
dependency should be easy to remove though by using any of the bundle methods
for SSVMs present in Shogun.

There are a couple of subgradient methods implemented in the graph directory.
Using this, you do not need Mosek. I have only tested them in the
graph learning task, but in principle they should work fine for label
sequence learning as well.

Apart from Shogun dependencies, you need cvxopt for the linear programming
relaxation used to solve the argmax in graph learning.

Report
======

You can get a copy of my final degree project (PFC in Spanish) report here.
[This version](https://dl.dropboxusercontent.com/u/11020840/pfc/fjig_pfc_bw.pdf)
is more appropriate for black & white printing
since the cross-links in the document are black. If you just want to read it
using a computer or similar, use beter this other
[version](https://dl.dropboxusercontent.com/u/11020840/pfc/fjig_pfc.pdf).

[Slides](https://www.dropbox.com/s/a0bolzzxgr669sg/fjig_pfc_slides.pdf)
in Spanish for the presentation. I plan to prepare something similar
in English soon for a workshop taking place in a few weeks. I will upload then
the source code as well.

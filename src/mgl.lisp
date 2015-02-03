(in-package :mgl)

(defsection @mgl-manual (:title "MGL Manual")
  (mgl asdf:system)
  (@mgl-overview section)
  (@mgl-dataset section)
  (@mgl-resample section)
  (@mgl-model section)
  (@mgl-monitoring section)
  (@mgl-classification section)
  (@mgl-opt section)
  (@mgl-diffun section)
  (@mgl-bp section)
  (@mgl-bm section)
  (@mgl-gp section)
  (@mgl-nlp section))

(defsection @mgl-overview (:title "Overview")
  "\\MGL is a Common Lisp machine learning library by [Gábor
  Melis](http://quotenil.com) with some parts originally contributed
  by Ravenpack International. It mainly concentrates on various forms
  of neural networks (boltzmann machines, feed-forward and recurrent
  backprop nets). Most of \\MGL is built on top of MGL-MAT so it has
  BLAS and CUDA support.

  In general, the focus is on power and performance not on ease of
  use. Perhaps one day there will be a cookie cutter interface with
  restricted functionality if a reasonable compromise is found between
  power and utility.

  Here is the [official repository](https://github.com/melisgl/mgl)
  and [HTML
  documentation](http://melisgl.github.io/mgl-pax-world/mgl-manual.html)."
  (@mgl-dependencies section)
  (@mgl-code-organization section)
  (@mgl-glossary section))

(defsection @mgl-dependencies (:title "Dependencies")
  "\\MGL used to rely on [LLA](https://github.com/tpapp/lla) to
  interface to BLAS and LAPACK. That's mostly history by now, but
  configuration of foreign libraries is still done via LLA. See the
  README in LLA on how to set things up. Note that these days OpenBLAS
  is easier to set up and just as fast as ATLAS.

  [CL-CUDA](https://github.com/takagi/cl-cuda) and
  [MGL-MAT](https://github.com/melisgl/mgl) are the two main
  dependencies and also the ones not yet in quicklisp, so just drop
  them into `quicklisp/local-projects/`. If there is no suitable GPU
  on the system or the CUDA SDK is not installed, \\MGL will simply
  fall back on using BLAS and Lisp code. Wrapping code in
  MGL-MAT:WITH-CUDA* is basically all that's needed to run on the GPU,
  and with MGL-MAT:CUDA-AVAILABLE-P one can check whether the GPU is
  really being used.

  Prettier-than-markdown HTML documentation cross-linked with other
  libraries is
  [available](http://melisgl.github.io/mgl-pax-world/mgl-manual.html)
  as part of [PAX World](http://melisgl.github.io/mgl-pax-world/).")

(defsection @mgl-code-organization (:title "Code Organization")
  "\\MGL consists of several packages dedicated to different tasks.
  For example, package `MGL-RESAMPLE` is about @MGL-RESAMPLE and
  `MGL-GD` is about @MGL-GD and so on. On one hand, having many
  packages makes it easier to cleanly separate API and implementation
  and also to explore into a specific task. At other times, they can
  be a hassle, so the MGL package itself reexports every external
  symbol found in all the other packages that make up \\MGL and
  \\MGL-MAT (see MGL-MAT:@MAT-MANUAL) on which it heavily relies.

  One exception to this rule is the bundled, but independent
  MGL-GNUPLOT library.

  The built in tests can be run with:

      (ASDF:OOS 'ASDF:TEST-OP '#:MGL)

  Note, that most of the tests are rather stochastic and can fail once
  in a while.")

(cl-reexport:reexport-from :mgl-mat)
(cl-reexport:reexport-from :mgl-common)
(cl-reexport:reexport-from :mgl-util)
(cl-reexport:reexport-from :mgl-log)
(cl-reexport:reexport-from :mgl-dataset)
(cl-reexport:reexport-from :mgl-resample)
(cl-reexport:reexport-from :mgl-core)
(cl-reexport:reexport-from :mgl-opt)
(cl-reexport:reexport-from :mgl-gd)
(cl-reexport:reexport-from :mgl-cg)
(cl-reexport:reexport-from :mgl-diffun)
(cl-reexport:reexport-from :mgl-bp)
(cl-reexport:reexport-from :mgl-bm)
(cl-reexport:reexport-from :mgl-unroll)
(cl-reexport:reexport-from :mgl-gp)
(cl-reexport:reexport-from :mgl-nlp)

(defsection @mgl-glossary (:title "Glossary")
  "Ultimately machine learning is about creating **models** of some
  domain. The observations in the modelled domain are called
  **instances** (also known as examples or samples). Sets of instances
  are called **datasets**. Datasets are used when fitting a model or
  when making **predictions**. Sometimes the word predictions is too
  specific, and the results obtained from applying a model to some
  instances are simply called **results**.")

(defsection @mgl-bm (:title "Boltzmann Machines"))

(defsection @mgl-gp (:title "Gaussian Processes"))

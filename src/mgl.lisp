(in-package :mgl)

(defsection @mgl-manual (:title "MGL Manual")
  (mgl asdf:system)
  (@mgl-overview section)
  (@mgl-dataset section)
  (@mgl-resample section)
  (@mgl-model section)
  (@mgl-opt section)
  (@mgl-diffun section)
  (@mgl-bp section)
  (@mgl-bm section)
  (@mgl-gp section))

(defsection @mgl-overview (:title "Overview")
  "\\MGL is a Common Lisp machine learning library by [Gábor
  Melis](http://quotenil.com) with some parts originally contributed
  by Ravenpack International. It implements:

  - Gradient descent optimization

      - Nesterov momentum
  - Conjugate gradient optimization
  - Backpropagation networks (\\BPN)

      - Dropout
      - Rectified linear units
      - Maxout
      - Max-channel
  - Boltzmann Machines

      - Restricted Boltzmann Machines (\\RBM)
      - Contrastive Divergence (CD) learning
      - Deep Belief Networks (\\DBN)
      - Semi Restricted Boltzmann Machines
      - Deep Boltzmann Machines
      - Persistent Contrastive Divergence (\\PCD) learning
      - Unrolling \\DBN or a \\DBM to a \\BPN
  - Gaussian Processes

      - Optimizing Gaussian Processes as BPNs

  In general, the focus is on power and performance not on ease of
  use. Perhaps one day there will be a cookie cutter interface with
  restricted functionality if a reasonable compromise is found between
  power and utility."
  (@mgl-dependencies section)
  (@mgl-code-organization section))

(defsection @mgl-dependencies (:title "Dependencies")
  "\\MGL used to rely on [LLA](https://github.com/tpapp/lla) to
  interface to BLAS and LAPACK. That's mostly history by now, but
  configuration of foreign libraries is still done via LLA. See the
  README in LLA on how to set things up. Note that these days OpenBLAS
  is easier to set up and just as fast as ATLAS.

  [CL-CUDA](https://github.com/takagi/cl-cuda) is a dependency for
  which the NVIDIA CUDA Toolkit needs to be installed, but \\MGL is
  fully functional even if there is no cuda capable gpu installed. See
  the MGL-MAT:WITH-CUDA* macro for how to use it.")

(defsection @mgl-code-organization (:title "Code Organization")
  "\\MGL consists of several packages dedicated to different tasks.
  For example, package `MGL-RESAMPLE` is about @MGL-RESAMPLE and
  `MGL-GD` is about @MGL-GD and so on. On one hand, having many
  packages makes it easier to cleanly separate API and implementation
  and also to explore into a specific task. At other times, they can
  be a hassle, so the `MGL` package itself reexports every external
  symbol found in all the other packages that make up \\MGL.

  One exception to this rule is the bundled, but independent
  MGL-GNUPLOT library.

  The built in tests can be run with:

      (ASDF:OOS 'ASDF:TEST-OP '#:MGL)

  Note, that most of the tests are rather stochastic and can fail once
  in a while.")

(cl-reexport:reexport-from :mgl-common)
(cl-reexport:reexport-from :mgl-util)
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

(defsection @mgl-bp (:title "Backprogation Neural Networks"))

(defsection @mgl-bm (:title "Boltzmann Machines"))

(defsection @mgl-gp (:title "Gaussian Processes"))

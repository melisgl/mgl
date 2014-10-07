(in-package :mgl)

(defsection @mgl-manual (:title "MGL Manual")
  (mgl asdf:system)
  (@mgl-overview section)
  (@mgl-basic-concepts section)
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

  - Backpropagation networks (\\BPN)

      - Dropout
      - Rectified linear units
      - Maxout
      - Max-channel
  - Boltzmann Machines
    - Restricted Boltzmann Machines (\\RBM)
    - Deep Belief Networks (\\DBN)
    - Semi Restricted Boltzmann Machines
    - Boltzmann Machines
    - Unrolling \\DBN to a \\BPN
    - Contrastive Divergence (CD) learning
    - Persistent Contrastive Divergence (\\PCD) learning
  - Gradient descent optimization
    - Nesterov momentum
  - Conjugate gradient optimization
  - Gaussian Processes
    - Optimizing Gaussian Processes as BPNs"
  (@mgl-features section)
  (@mgl-dependencies section)
  (@mgl-tests section)
  (@mgl-bundled-software section))

(defsection @mgl-features (:title "Features")
  "In general, the focus is on power and performance not on ease of use.
  For example, it's possible to:

  - control the order of presentation of training examples,
  - vary learning rate depending on time, state of the optimizer,
  - track all kinds of statistics during training,
  etc.

  Perhaps one day there will be a cookie cutter interface with
  restricted functionality if a reasonable compromise is found between
  power and utility.")

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

(defsection @mgl-tests (:title "Tests")
  "Run the built in tests 
  with:

      (ASDF:OOS 'ASDF:TEST-OP '#:MGL)

  Note, that most of the tests are rather stochastic and can fail once
  in a while.")

(defsection @mgl-bundled-software (:title "Bundled Software")
  "With [MGL-PAX](https://github.com/melisgl/mgl-pax) and
  [MGL-MAT](https://github.com/melisgl/mgl-mat) libraries split off
  there remains only a single library bundled with \\MGL which does
  not depend on the rest of \\MGL:

  - MGL-GNUPLOT, a plotting library.

  There is also MGL-VISUALS which does depend on \\MGL.")

(defsection @mgl-basic-concepts (:title "Basic Concepts")
  "MODEL, training set, test set, validation set,
  sample/instance/example.")

(defsection @mgl-bp (:title "Backprogation Neural Networks"))

(defsection @mgl-bm (:title "Boltzmann Machines"))

(defsection @mgl-gp (:title "Gaussian Processes"))

(mgl-pax:define-package #:mgl-common
  (:documentation "The only purpose of this package is to avoid
  conflicts between other packages.")
  (:use :common-lisp :mgl-pax)
  (:export #:name #:name= #:default-value #:cost #:size #:nodes #:group-size
           #:target #:fn #:weights))

(cl:defpackage #:mgl-util
  (:use #:common-lisp #:mgl-mat #:mgl-common)
  (:export
   #:name=
   ;; Macrology
   #:special-case
   #:apply-key
   ;; Types
   #:flt
   #:flt-ctype
   #:positive-flt
   #:most-negative-flt
   #:least-negative-flt
   #:least-positive-flt
   #:most-positive-flt
   #:flt-vector
   #:index
   #:index-vector
   #:*no-array-bounds-check*
   #:the!
   #:with-zero-on-underflow
   ;; Pathnames
   #:asdf-system-relative-pathname
   ;; Misc
   #:split-plist
   #:while
   #:last1
   #:append1
   #:push-all
   #:group
   #:subseq*
   #:max-position
   #:hash-table->vector
   #:repeatedly
   #:make-sequence-generator
   #:make-random-generator
   #:applies-to-p
   #:uninterned-symbol-p
   ;; Periodic fn
   #:periodic-fn
   #:call-periodic-fn
   #:call-periodic-fn!
   #:last-eval
   ;; Math
   #:sign
   #:sech
   #:sigmoid
   #:scaled-tanh
   #:half-life-to-decay
   #:half-life-to-decay-rate
   #:decay-to-half-life
   #:decay-rate-to-half-life
   #:log-prob-to-perplexity
   #:perplexity-to-log-prop
   #:try-chance
   #:binarize-randomly
   #:gaussian-random-1
   #:poisson-random
   #:select-random-element
   #:binomial-log-likelihood-ratio
   #:multinomial-log-likelihood-ratio
   ;; Running stat
   #:running-stat
   #:clear-running-stat
   #:add-to-running-stat
   #:running-stat-variance
   #:running-stat-mean
   ;; Permutations
   #:permute
   #:invert-permutation
   #:sorting-permutation
   ;; Array utilities
   #:as-column-vector
   #:rows-to-arrays
   #:max-row-positions
   ;; Classes
   #:defclass-now
   #:defmaker
   ;; Printing
   #:print-table
   ;; Describe customization
   #:define-descriptions
   ;; Copy
   #:with-copying
   #:copy-object-extra-initargs
   #:copy-object-slot
   #:define-slots-not-to-be-copied
   #:define-slots-to-be-shallow-copied
   #:copy
   ;; Feature selection, encoding
   #:count-features
   #:compute-feature-llrs
   #:compute-feature-disambiguities
   #:index-scored-features
   #:read-indexed-features
   #:write-indexed-features
   #:encode/bag-of-words
   ;; Repeatable experiments
   #:*experiment-random-seed*
   #:call-repeatably
   #:repeatably)
  (:documentation "Simple utilities, types."))

(mgl-pax:define-package #:mgl-log
  (:documentation "See MGL-LOG:@MGL-LOG.")
  (:use #:common-lisp #:mgl-pax #:mgl-common #:mgl-util))

(mgl-pax:define-package #:mgl-dataset
  (:documentation "See MGL-DATASET:@MGL-DATASET.")
  (:use #:common-lisp #:mgl-pax #:mgl-common #:mgl-util))

(mgl-pax:define-package #:mgl-resample
  (:documentation "See MGL-RESAMPLE:@MGL-RESAMPLE.")
  (:use #:common-lisp #:mgl-pax))

(mgl-pax:define-package #:mgl-core
  (:use #:common-lisp #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-log
        #:mgl-dataset)
  (:documentation "See MGL-CORE:@MGL-MODEL, MGL-CORE:@MGL-MONITOR,
  MGL-CORE:@MGL-CLASSIFICATION."))

(mgl-pax:define-package #:mgl-opt
  (:documentation "See MGL-OPT:@MGL-OPT.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-log
        #:mgl-dataset #:mgl-core))

(mgl-pax:define-package #:mgl-gd
  (:documentation "See MGL-GD:@MGL-GD.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util
        #:mgl-dataset #:mgl-core
        #:mgl-opt)
  (:export #:@mgl-gd))

(mgl-pax:define-package #:mgl-cg
  (:documentation "See MGL-CG:@MGL-CG.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-log
        #:mgl-dataset #:mgl-core
        #:mgl-opt)
  (:export #:@mgl-cg))

(mgl-pax:define-package #:mgl-diffun
  (:documentation "See MGL-DIFFUN:@MGL-DIFFUN.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-core
        #:mgl-opt))

(mgl-pax:define-package #:mgl-bp
  (:documentation "See MGL-BP:@MGL-BP.")
  (:use #:common-lisp #:cl-cuda #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util
        #:mgl-dataset #:mgl-core
        #:mgl-opt #:mgl-gd #:mgl-cg))

(cl:defpackage #:mgl-bm
  (:use #:common-lisp #:cl-cuda #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-core
        #:mgl-opt #:mgl-gd)
  (:nicknames #:mgl-rbm)
  (:export
   ;; Chunk
   #:name
   #:size
   #:chunk
   #:chunk-size
   #:inputs
   #:nodes
   #:means
   #:indices-present
   #:constant-chunk
   #:default-value
   #:conditioning-chunk
   #:sigmoid-chunk
   #:gaussian-chunk
   #:relu-chunk
   #:scale
   #:group-size
   #:exp-normalized-group-chunk
   #:softmax-chunk
   #:constrained-poisson-chunk
   #:temporal-chunk
   ;; Chunk extensions
   #:set-chunk-mean
   #:sample-chunk
   ;; Cloud
   #:cloud
   #:name
   #:chunk1
   #:chunk2
   #:conditioning-cloud-p
   #:cloud-chunk-among-chunks
   #:full-cloud
   #:weights
   #:factored-cloud
   #:cloud-a
   #:cloud-b
   #:rank
   ;; BM
   #:bm
   #:chunks
   #:visible-chunks
   #:hidden-chunks
   #:conditioning-chunks
   #:find-chunk
   #:merge-cloud-specs
   #:clouds
   #:do-clouds
   #:find-cloud
   #:importances
   ;; Operating an BM
   #:set-visible-mean/1
   #:set-hidden-mean/1
   #:sample-visible
   #:sample-hidden
   ;; Mean field
   #:supervise-mean-field/default
   #:default-mean-field-supervisor
   #:settle-mean-field
   #:settle-visible-mean-field
   #:settle-hidden-mean-field
   #:set-visible-mean
   #:set-hidden-mean
   ;; DBM
   #:dbm
   #:layers
   #:clouds-up-to-layers
   #:up-dbm
   #:down-dbm
   #:dbm->dbn
   ;; RBM
   #:rbm
   #:dbn
   ;; Sparsity
   #:sparser
   #:sparsity-gradient-source
   #:normal-sparsity-gradient-source
   #:cheating-sparsity-gradient-source
   #:target
   #:cost
   #:damping
   ;; Stuff common to learners
   #:visible-sampling
   #:hidden-sampling
   #:n-gibbs
   #:positive-phase
   #:negative-phase
   #:bm-learner
   ;; Contrastive Divergence (CD) learning for RBMs
   #:rbm-cd-learner
   ;; Persistent Contrastive Divergence (PCD) learning
   #:bm-pcd-learner
   #:n-particles
   #:persistent-chains
   #:pcd
   ;; Convenience, utilities
   #:inputs->nodes
   #:nodes->inputs
   #:reconstruction-rmse
   #:reconstruction-error
   #:monitor-bm-mean-field-bottom-up
   #:monitor-bm-mean-field-reconstructions
   #:make-reconstruction-monitors
   #:make-reconstruction-monitors*
   ;; Classification
   #:softmax-label-chunk
   ;; DBN
   #:dbn
   #:rbms
   #:n-rbms
   #:down-mean-field
   #:monitor-dbn-mean-field-bottom-up
   #:monitor-dbn-mean-field-reconstructions
   #:mark-everything-present)
  (:documentation "Fully General Boltzmann Machines, Restricted
  Boltzmann Machines and their stacks called Deep Belief
  Networks (DBN)."))

(cl:defpackage #:mgl-unroll
  (:use #:common-lisp #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-dataset #:mgl-core
        #:mgl-bm #:mgl-bp)
  (:export
   #:chunk-lump-name
   #:unroll-dbn
   #:unroll-dbm
   #:initialize-fnn-from-bm
   ;; BPN setup
   #:set-dropout-and-rescale-activation-weights
   ;; SET-INPUT support for BPN converted from a DBM with MAP lumps
   #:fnn-clamping-cache
   #:clamping-cache
   #:populate-key
   #:populate-map-cache-lazily-from-dbm
   #:populate-map-cache)
  (:documentation "Translating Boltzmann Machines to a Backprop
  networks, aka `unrolling'."))

(cl:defpackage #:mgl-gp
  (:use #:common-lisp #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-core
        #:mgl-bp)
  (:export
   #:gp
   #:gp-means
   #:gp-covariances
   #:gp-means-and-covariances
   #:gp-means-and-covariances*
   ;;
   #:prior-gp
   #:posterior-gp
   #:update-gp
   ;; BPN-GP
   #:fnn-gp
   #:mean-lump-name
   #:covariance-lump-name
   #:->gp
   #:->ref
   #:->rep
   #:->stretch
   #:->rough-exponential
   #:->periodic)
  (:export
   #:gp-confidences-as-plot-data
   #:gp-samples-as-plot-data)
  (:documentation "Gaussian processes with support for training with
  backpropagation."))

(mgl-pax:define-package #:mgl-nlp
  (:documentation "See MGL-NLP:@MGL-NLP.")
  (:use #:common-lisp #:mgl-pax #:mgl-util))

(mgl-pax:define-package #:mgl
  (:documentation "See MGL:@MGL-MANUAL. This package reexports
  everything from other packages defined here plus MGL-MAT.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat
        #:mgl-common #:mgl-util #:mgl-log
        #:mgl-dataset #:mgl-resample #:mgl-core
        #:mgl-opt #:mgl-gd #:mgl-cg
        #:mgl-diffun #:mgl-bp #:mgl-bm #:mgl-unroll #:mgl-gp
        #:mgl-nlp))

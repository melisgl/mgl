(mgl-pax:define-package :mgl-common
  (:use :common-lisp :mgl-pax)
  ;; FIXME: remove these after everything is defined with
  ;; MGL-PAX:DEFINE-PACKAGE.
  (:export :name :name= :default-value :cost :size :nodes :group-size :target))

(mgl-pax:define-package :mgl
  (:documentation "See MGL:@MGL-MANUAL. This package reexports
  everything from other packages defined here plus MGL-MAT.")
  (:use :common-lisp :mgl-pax))

(mgl-pax:define-package :mgl-resample
  (:documentation "See MGL-RESAMPLE:@MGL-RESAMPLE.")
  (:use :common-lisp :mgl-pax))

(cl:defpackage :mgl-util
  (:use #:common-lisp #:mgl-mat #:mgl-common)
  (:export
   #:name=
   ;; Macrology
   #:special-case
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
   #:make-seq-generator
   #:make-random-generator
   #:make-n-gram-mappee
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
   ;; Array utilities
   #:as-column-vector
   ;; I/O
   #:write-weights
   #:read-weights
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
   ;; Confusion matrix
   #:confusion-matrix
   #:sort-confusion-classes
   #:confusion-class-name
   #:confusion-count
   #:map-confusion-matrix
   #:confusion-matrix-classes
   #:confusion-matrix-accuracy
   #:confusion-matrix-precision
   #:confusion-matrix-recall
   #:add-confusion-matrix
   ;; Feature selection, encoding
   #:count-features
   #:compute-feature-llrs
   #:compute-feature-disambiguities
   #:index-scored-features
   #:read-indexed-features
   #:write-indexed-features
   #:encode/bag-of-words)
  (:documentation "Simple utilities, types."))

(mgl-pax:define-package :mgl-dataset
  (:documentation "See MGL-DATASET:@MGL-DATASET.")
  (:use #:common-lisp #:mgl-pax #:mgl-common #:mgl-util)
  (:export #:@mgl-dataset))

(cl:defpackage :mgl-core
  (:use #:common-lisp #:mgl-mat #:mgl-common #:mgl-util #:mgl-dataset)
  (:export
   #:set-input
   ;; Error counter
   #:counter
   #:print-counter
   #:error-counter
   #:misclassification-counter
   #:cross-entropy-counter
   #:rmse-counter
   #:sum-errors
   #:n-sum-errors
   #:add-error
   #:reset-counter
   #:get-error
   ;; Stripes
   #:n-stripes
   #:set-n-stripes
   #:max-n-stripes
   #:set-max-n-stripes
   #:find-striped
   #:striped-array
   #:stripe-start
   #:stripe-end
   #:with-stripes
   ;; Collecting errors
   #:map-batches-for-learner
   #:do-batches-for-learner
   #:apply-counters-and-measurers
   #:collect-batch-errors
   ;; Executors
   #:map-over-executors
   #:do-executors
   #:find-one-executor
   #:trivial-cached-executor-mixin
   #:lookup-executor-cache
   #:insert-into-executor-cache
   #:trivially-map-over-executors
   #:sample-to-executor-cache-key
   ;; Segments
   #:map-segments
   #:segment-weights
   #:with-segment-weights
   #:map-segment-runs
   #:list-segments
   ;; Segment set
   #:segment-set
   #:segments
   #:start-indices
   #:do-segment-set
   #:segment-set-size
   #:segment-set->mat
   #:segment-set<-mat
   ;; Classification
   #:label
   #:label-distribution
   #:labeled
   #:labeledp
   #:stripe-label
   #:maybe-make-misclassification-measurer
   #:classification-confidences
   #:maybe-make-cross-entropy-measurer
   #:roc-auc
   #:roc-auc-counter)
  (:documentation "Generic training related interfaces and basic
  definitions. The three most important concepts are SAMPLERs,
  TRAINERs and LEARNERs."))

(mgl-pax:define-package :mgl-opt
  (:documentation "See MGL-OPT:@MGL-OPT.")
  (:use #:common-lisp #:mgl-pax #:mgl-common #:mgl-dataset #:mgl-core))

(mgl-pax:define-package :mgl-diffun
  (:documentation "See MGL-DIFFUN:@MGL-DIFFUN.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat #:mgl-common #:mgl-util #:mgl-core
        #:mgl-opt))

(mgl-pax:define-package :mgl-gd
  (:documentation "See MGL-GD:@MGL-GD.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat #:mgl-common #:mgl-util :mgl-dataset
        #:mgl-core #:mgl-opt)
  (:export #:@mgl-gd))

(mgl-pax:define-package :mgl-cg
  (:documentation "See MGL-CG:@MGL-CG.")
  (:use #:common-lisp #:mgl-pax #:mgl-mat #:mgl-common #:mgl-util :mgl-dataset
        #:mgl-core #:mgl-opt)
  (:export #:@mgl-cg))

(cl:defpackage :mgl-bm
  (:use #:common-lisp #:cl-cuda #:mgl-mat #:mgl-common #:mgl-util #:mgl-core
        ;; #:mgl-dataset
        #:mgl-opt #:mgl-gd #:mgl-cg)
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
   #:make-bm-reconstruction-rmse-counters-and-measurers
   #:make-dbm-reconstruction-rmse-counters-and-measurers
   #:collect-bm-mean-field-errors
   ;; Classification
   #:softmax-label-chunk
   #:make-bm-reconstruction-misclassification-counters-and-measurers
   #:make-bm-reconstruction-cross-entropy-counters-and-measurers
   #:collect-bm-mean-field-errors/labeled
   ;; DBN
   #:dbn
   #:rbms
   #:down-mean-field
   #:make-dbn-reconstruction-rmse-counters-and-measurers
   #:collect-dbn-mean-field-errors
   #:mark-labels-present
   #:mark-everything-present
   #:make-dbn-reconstruction-misclassification-counters-and-measurers
   #:collect-dbn-mean-field-errors/labeled)
  (:documentation "Fully General Boltzmann Machines, Restricted
  Boltzmann Machines and their stacks called Deep Belief
  Networks (DBN)."))

(cl:defpackage :mgl-bp
  (:use #:common-lisp #:cl-cuda #:mgl-mat #:mgl-common #:mgl-util ;; #:mgl-dataset
        #:mgl-core #:mgl-opt #:mgl-gd #:mgl-cg)
  (:export
   #:lump
   #:deflump
   #:name
   #:size
   #:default-size
   #:lump-size
   #:lump-node-array
   #:->input
   #:update-stats-p
   #:normalize-with-stats-p
   #:normalized-cap
   #:->weight
   #:->constant
   #:default-value
   #:->normalized
   #:group-size
   #:->activation
   #:transpose-weights-p
   #:add-activations
   #:->error
   #:importance
   #:cost
   #:transfer-lump
   #:derive-lump
   ;; BPN
   #:bpn
   #:nodes
   #:derivatives
   #:lumps
   #:find-lump
   #:initialize-lump
   #:initialize-bpn
   #:add-lump
   #:remove-lump
   #:with-weights-copied
   #:build-bpn
   #:forward-bpn
   #:backward-bpn
   #:bp-learner
   #:compute-derivatives
   #:dropout
   ;; Node types
   #:define-node-type
   #:node
   #:add-derivative
   #:->rep
   #:->stretch
   #:->+
   #:->*
   #:->sum
   #:->linear
   #:->sigmoid
   #:->scaled-tanh
   #:->rectified
   #:->split-sign
   #:noisyp
   #:->dropout
   #:->multiply-with-gaussian
   #:->sample-binary
   #:->softplus
   #:->exp
   #:->abs
   #:->sin
   #:->rough-exponential
   #:->ref
   #:->periodic
   #:->sum-squared-error
   #:->squared-error
   #:->max
   #:->max-channel
   #:->min
   #:->softmax
   #:->cross-entropy
   #:->cross-entropy-softmax
   #:softmax
   #:target
   #:class-weights
   #:add-cross-entropy-softmax
   ;; Utilities
   #:collect-bpn-errors
   #:renormalize-activations
   #:arrange-for-renormalizing-activations)
  (:documentation "Backpropagation."))

(cl:defpackage :mgl-unroll
  (:use #:common-lisp #:mgl-mat #:mgl-util #:mgl-dataset #:mgl-core
        #:mgl-bm #:mgl-bp #:mgl-gd)
  (:export
   #:chunk-lump-name
   #:unroll-dbn
   #:unroll-dbm
   #:initialize-bpn-from-bm
   ;; BPN setup
   #:set-dropout-and-rescale-activation-weights
   ;; SET-INPUT support for BPN converted from a DBM with MAP lumps
   #:bpn-clamping-cache
   #:clamping-cache
   #:populate-key
   #:populate-map-cache-lazily-from-dbm
   #:populate-map-cache)
  (:documentation "Translating Boltzmann Machines to a Backprop
  networks, aka `unrolling'."))

(cl:defpackage :mgl-gp
  (:use #:common-lisp #:mgl-mat #:mgl-common #:mgl-util #:mgl-core #:mgl-bp)
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
   #:bpn-gp
   #:mean-lump-name
   #:covariance-lump-name
   #:->gp)
  (:export
   #:gp-confidences-as-plot-data
   #:gp-samples-as-plot-data)
  (:documentation "Gaussian processes with support for training with
  backpropagation."))

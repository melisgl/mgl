(cl:defpackage :mgl-util
  (:use #:common-lisp)
  (:export
   ;; Macrology
   #:with-gensyms
   #:split-body
   #:suffix-symbol
   #:special-case
   ;; Types
   #:flt
   #:positive-flt
   #:flt-vector
   #:make-flt-array
   #:index
   #:index-vector
   ;; Declarations
   #:*no-array-bounds-check*
   #:the!
   ;; Misc
   #:split-plist
   #:while
   #:last1
   #:append1
   #:push-all
   #:group
   #:subseq*
   #:hash-table->alist
   #:alist->hash-table
   #:hash-table->vector
   #:reverse-hash-table
   #:repeatedly
   #:nshuffle-vector
   #:make-random-generator
   #:make-n-gram-mappee
   #:break-seq
   #:stratified-split
   ;; Math
   #:sigmoid
   #:try-chance
   #:binarize-randomly
   #:gaussian-random-1
   #:poisson-random
   #:select-random-element
   #:binomial-log-likelihood-ratio
   #:multinomial-log-likelihood-ratio
   ;; Blas support
   #:*use-blas*
   #:use-blas-p
   #:cost-of-copy
   #:cost-of-fill
   #:cost-of-gemm
   #:storage
   #:reshape2
   #:set-ncols
   #:sum-elements
   ;; I/O 
   #:read-single-float-vector
   #:write-single-float-vector
   #:read-double-float-vector
   #:write-double-float-vector
   #:write-weights
   #:read-weights
   ;; Printing
   #:print-table
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
   #:index-scored-features
   #:read-indexed-features
   #:write-indexed-features
   #:encode/bag-of-words)
  (:documentation "Simple utilities, types."))

(cl:defpackage :mgl-train
  (:use #:common-lisp #:mgl-util)
  (:export
   #:train
   #:train-batch
   #:set-input
   #:initialize-trainer
   #:n-inputs-until-update
   ;; Sampler
   #:sample
   #:finishedp
   #:function-sampler
   #:sampler
   #:counting-sampler
   #:n-samples
   #:max-n-samples
   #:counting-function-sampler
   #:sample-batch
   ;; Error counter
   #:error-counter
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
   #:stripe-start
   #:stripe-end
   #:with-stripes
   ;; Collecting errors
   #:map-batches-for-learner
   #:do-batches-for-learner
   #:collect-batch-errors
   ;; Segments
   #:map-segments
   #:segment-weights
   #:with-segment-weights
   #:segment-size
   #:map-segment-runs
   #:list-segments
   ;; Segment set
   #:segment-set
   #:segments
   #:start-indices
   #:do-segment-set
   #:segment-set-size
   #:segment-set->weights
   #:segment-set<-weights
   ;; Common generic functions
   #:name
   #:size
   #:nodes
   #:default-value
   #:group-size
   #:batch-size
   #:n-inputs
   #:label)
  (:documentation "Generic training related interfaces and basic
definitions. The three most important concepts are SAMPLERs, TRAINERs
and LEARNERs."))

(cl:defpackage :mgl-gd
  (:use #:common-lisp #:mgl-util #:mgl-train)
  (:export
   #:map-segment-gradient-accumulators
   #:do-segment-gradient-accumulators
   #:find-segment-gradient-accumulator
   #:with-segment-gradient-accumulator
   #:maybe-update-weights
   #:update-weights
   ;; Gradient descent
   #:gd-trainer
   #:n-inputs
   #:use-accumulator2
   #:accumulator
   #:learning-rate
   #:momentum
   #:weight-decay
   #:weight-penalty
   #:batch-size
   #:batch-gd-trainer
   #:n-inputs-in-batch
   #:before-update-hook
   #:normalized-batch-gd-trainer
   #:per-weight-batch-gd-trainer
   #:n-weight-uses-in-batch
   ;; Segmented trainer
   #:segmented-gd-trainer
   #:segmenter
   #:trainers
   #:n-inputs
   #:find-trainer-for-segment)
  (:documentation "Generic, gradient based optimization related
interface and simple gradient descent based trainers."))

(cl:defpackage :mgl-cg
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd)
  (:export
   #:cg
   #:*default-int*
   #:*default-ext*
   #:*default-sig*
   #:*default-rho*
   #:*default-ratio*
   #:*default-max-n-line-searches*
   #:*default-max-n-evaluations-per-line-search*
   #:*default-max-n-evaluations*
   #:cg-trainer
   #:cg-args
   #:n-inputs
   #:segment-set
   #:accumulator
   #:compute-batch-cost-and-derive
   #:decayed-cg-trainer-mixin)
  (:documentation "Conjugate gradient based trainer."))

(cl:defpackage :mgl-bm
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd #:mgl-cg)
  (:nicknames #:mgl-rbm)
  (:export
   ;; Chunk
   #:name
   #:chunk
   #:chunk-size
   #:inputs
   #:nodes
   #:indices-present
   #:constant-chunk
   #:default-value
   #:conditioning-chunk
   #:sigmoid-chunk
   #:gaussian-chunk
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
   #:find-chunk
   #:merge-cloud-specs
   #:clouds
   #:do-clouds
   #:find-cloud
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
   ;; Stuff common to trainers
   #:visible-sampling
   #:hidden-sampling
   #:n-gibbs
   #:positive-phase
   #:negative-phase
   ;; Contrastive Divergence (CD) learning for RBMs
   #:rbm-cd-trainer
   ;; Persistent Contrastive Divergence (PCD) learning
   #:bm-pcd-trainer
   #:n-particles
   #:persistent-chains
   #:pcd
   ;; Convenience, utilities
   #:inputs->nodes
   #:nodes->inputs
   #:reconstruction-rmse
   #:reconstruction-error
   #:make-bm-reconstruction-rmse-counters-and-measurers
   #:bm-mean-field-errors
   ;; DBN
   #:dbn
   #:rbms
   #:down-mean-field
   #:make-dbn-reconstruction-rmse-counters-and-measurers
   #:dbn-mean-field-errors)
  (:documentation "Fully General Boltzmann Machines, Restricted
Boltzmann Machines and their stacks called Deep Belief
Networks (DBN)."))

(cl:defpackage :mgl-bp
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd #:mgl-cg)
  (:export
   #:lump
   #:name
   #:lump-size
   #:lump-node-array
   #:indices-to-calculate
   #:input-lump
   #:weight-lump
   #:constant-lump
   #:default-value
   #:normalized-lump
   #:group-size
   #:activation-lump
   #:transpose-weights-p
   #:error-node
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
   #:build-bpn
   #:forward-bpn
   #:backward-bpn
   #:bp-trainer
   #:compute-derivatives
   #:cg-bp-trainer
   ;; Node types
   #:define-node-type
   #:node
   #:add-derivative
   #:->+
   #:->sum
   #:->linear
   #:->sigmoid
   #:->exp
   #:->sum-squared-error
   #:->cross-entropy
   #:cross-entropy-softmax-lump
   #:softmax
   #:target)
  (:documentation "Backpropagation."))

(cl:defpackage :mgl-unroll
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-bm #:mgl-bp #:mgl-gd)
  (:export
   #:chunk-lump-name
   #:unroll-dbn
   #:unroll-dbm
   #:initialize-bpn-from-bm)
  (:documentation "Translating Boltzmann Machines to a Backprop
networks, aka `unrolling'."))

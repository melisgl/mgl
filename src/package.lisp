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
   #:most-negative-flt
   #:least-negative-flt
   #:least-positive-flt
   #:most-positive-flt
   #:flt-vector
   #:flt-matrix
   #:make-flt-array
   #:index
   #:index-vector
   ;; Declarations
   #:*no-array-bounds-check*
   #:the!
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
   #:hash-table->alist
   #:alist->hash-table
   #:hash-table->vector
   #:reverse-hash-table
   #:repeatedly
   #:nshuffle-vector
   #:shuffle-vector
   #:make-seq-generator
   #:make-random-generator
   #:make-n-gram-mappee
   #:break-seq
   #:stratified-split
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
   #:mv-gaussian-random
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
   #:backing-array
   #:fill!
   #:to-scalar
   #:as-column-vector
   ;; I/O
   #:read-double-float-array
   #:write-double-float-array
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
   #:batch-size
   #:cost
   #:group-size
   #:default-value
   #:label
   #:label-distribution
   #:n-inputs
   #:nodes
   #:name
   #:size
   #:target
   ;; Classification
   #:label
   #:labeled
   #:labeledp
   #:stripe-label
   #:maybe-make-misclassification-measurer
   #:classification-confidences
   #:maybe-make-cross-entropy-measurer
   #:roc-auc
   #:roc-auc-counter)
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
   #:after-update-hook
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
   #:make-dbn-reconstruction-misclassification-counters-and-measurers
   #:collect-dbn-mean-field-errors/labeled)
  (:documentation "Fully General Boltzmann Machines, Restricted
Boltzmann Machines and their stacks called Deep Belief
Networks (DBN)."))

(cl:defpackage :mgl-bp
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd #:mgl-cg)
  (:export
   #:lump
   #:deflump
   #:name
   #:size
   #:default-size
   #:lump-size
   #:lump-node-array
   #:indices-to-calculate
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
   #:bp-trainer
   #:compute-derivatives
   #:cg-bp-trainer
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
   #:->stochastic-sigmoid
   #:->scaled-tanh
   #:->rectified
   #:noisyp
   #:->softplus
   #:->exp
   #:->abs
   #:->rough-exponential
   #:->ref
   #:->periodic
   #:->sum-squared-error
   #:->squared-error
   #:->dropout
   #:->max
   #:->softmax
   #:->cross-entropy
   #:->cross-entropy-softmax
   #:softmax
   #:target
   #:class-weights
   ;; Utilities
   #:collect-bpn-errors
   #:renormalize-activations)
  (:documentation "Backpropagation."))

(cl:defpackage :mgl-unroll
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-bm #:mgl-bp #:mgl-gd)
  (:export
   #:chunk-lump-name
   #:unroll-dbn
   #:unroll-dbm
   #:initialize-bpn-from-bm
   ;; SET-INPUT support for BPN converted from a DBM with MAP lumps
   #:bpn-clamping-cache
   #:clamping-cache
   #:populate-key
   #:populate-map-cache-lazily-from-dbm
   #:populate-map-cache)
  (:documentation "Translating Boltzmann Machines to a Backprop
networks, aka `unrolling'."))

(cl:defpackage :mgl-gp
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-bp)
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

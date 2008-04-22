(cl:defpackage :mgl-util
  (:use #:common-lisp)
  (:export #:*use-blas*
           #:use-blas-p
           #:blas-supports-displaced-arrays-p
           #:with-gensyms
           #:split-body
           #:suffix-symbol
           #:special-case
           #:flt
           #:flt-vector
           #:make-flt-array
           #:index
           #:index-vector
           #:*no-array-bounds-check*
           #:*the*
           #:gaussian-random-1
           #:select-random-element
           #:split-plist
           #:last1
           #:push-all
           #:name
           #:value
           #:group-size
           #:repeatedly
           #:sigmoid
           #:try-chance
           #:binarize-randomly
           #:write-weights
           #:read-weights)
  (:documentation "Simple utilities, types, symbols of common
accessors such as NAME."))

(cl:defpackage :mgl-train
  (:use #:common-lisp #:mgl-util)
  (:export #:train
           #:train-one
           #:train-batch
           #:batch-trainer
           #:batch-size
           #:set-input
           #:initialize-trainer
           ;; Sampler
           #:sample
           #:finishedp
           #:function-sampler
           #:sampler
           #:counting-sampler
           #:n-samples
           #:max-n-samples
           #:counting-function-sampler
           ;; Error counter
           #:error-counter
           #:rmse-counter
           #:sum-errors
           #:n-sum-errors
           #:add-error
           #:reset-counter
           #:get-error
           ;; Segments
           #:map-segments
           #:segment-weights
           #:segment-derivatives
           #:with-segment-weights
           #:map-segment-runs
           #:supports-partial-updates-p
           #:list-segments
           ;; Segment set
           #:segment-set
           #:segments
           #:start-indices
           #:do-segment-set
           #:segment-set-size
           #:segment-set->weights
           #:segment-set<-weights)
  (:documentation "Generic training related interfaces and basic
definitions. The three most important concepts are SAMPLERs, TRAINERs
and LEARNERs."))

(cl:defpackage :mgl-gd
  (:use #:common-lisp #:mgl-util #:mgl-train)
  (:export #:map-segment-gradient-accumulators
           #:do-segment-gradient-accumulators
           #:maybe-update-weights
           ;; Gradient descent
           #:gd-trainer
           #:n-inputs
           #:use-accumulator2
           #:accumulator1
           #:accumulator2
           #:learning-rate
           #:momentum
           #:weight-decay
           #:batch-gd-trainer
           #:batch-size
           #:n-inputs-in-batch
           #:n-batches
           #:per-weight-batch-gd-trainer
           #:n-weight-uses-in-batch
           ;; Segmented trainer
           #:segmented-trainer
           #:segmenter
           #:trainers
           #:n-inputs)
  (:documentation "Generic, gradient based optimization related
interface and simple gradient descent based trainers."))

(cl:defpackage :mgl-cg
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd)
  (:export #:cg
           #:*default-int*
           #:*default-ext*
           #:*default-sig*
           #:*default-rho*
           #:*default-ratio*
           #:*default-max-n-line-searches*
           #:*default-max-n-evaluations-per-line-search*
           #:*default-max-n-evaluations*
           #:batch-cg-trainer
           #:cg-args
           #:compute-batch-cost-and-derive
           #:decayed-cg-trainer-mixin)
  (:documentation "Conjugate gradient based trainer."))

(cl:defpackage :mgl-rbm
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd #:mgl-cg)
  (:export #:rbm
           #:visible-chunks
           #:hidden-chunks
           #:default-clouds
           #:clouds
           #:do-clouds
           #:find-cloud
           ;; Operating an RBM
           #:set-visible-mean
           #:sample-visible
           #:set-hidden-mean
           #:sample-hidden
           ;; RBM trainer
           #:rbm-trainer
           #:sample-visible-p
           #:sample-hidden-p
           #:n-gibbs
           #:rmse-counting-rbm-trainer
           ;; Cloud
           #:cloud
           #:name
           #:visible-chunk
           #:hidden-chunk
           #:weights
           ;; Chunk
           #:name
           #:chunk
           #:chunk-size
           #:samples
           #:means
           #:indices-present
           #:constant-chunk
           #:value
           #:conditioning-chunk
           #:sigmoid-chunk
           #:gaussian-chunk
           #:scale
           #:group-size
           #:exp-normalized-group-chunk
           #:softmax-chunk
           #:constrained-poisson-chunk
           ;; Chunk extensions
           #:set-chunk-mean
           #:sample-chunk
           ;; DBN
           #:dbn
           #:rbms
           #:up-mean-field
           #:down-mean-field
           #:reconstruct-mean-field
           #:dbn-rmse
           ;;
           #:get-squared-error)
  (:documentation "Restricted Boltzmann Machines (RBM) and their
stacks called Deep Belief Networks (DBN)."))

(cl:defpackage :mgl-bp
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd #:mgl-cg)
  (:export #:lump
           #:name
           #:lump-size
           #:lump-node-array
           #:indices-to-calculate
           #:do-lump
           #:nodewise-lump
           #:input-lump
           #:weight-lump
           #:constant-lump
           #:value
           #:normalized-lump
           #:group-size
           #:hidden-lump
           #:output-lump
           #:cross-entropy-softmax-lump
           #:cross-entropy-softmax-error
           #:activation-lump
           #:transpose-weights-p
           #:error-node
           #:importance
           #:transfer-lump
           #:derivate-lump
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
           #:cg-bp-trainer
           ;; Node types
           #:define-node-type
           #:node
           #:add-derivative
           #:->+
           #:->linear
           #:->sigmoid
           #:->exp
           #:->sum-squared-error
           #:->cross-entropy
           #:ref
           #:sub
           #:col)
  (:documentation "Backpropagation."))

(cl:defpackage :mgl-unroll-dbn
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-rbm #:mgl-bp #:mgl-gd)
  (:export #:unroll-dbn
           #:clamp-indices-in-unrolled-dbn
           #:initialize-bpn-from-dbn)
  (:documentation "Translating a DBN to a backprop network, aka
`unrolling'."))

(cl:defpackage :mgl-svd
  (:use #:common-lisp #:mgl-util #:mgl-gd)
  (:export))

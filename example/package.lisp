(cl:defpackage :mgl-example-util
  (:use #:common-lisp #:mgl-util #:mgl-mat #:mgl-train #:mgl-gd #:mgl-bm
        #:mgl-bp)
  (:export #:*example-dir*
           #:time->string
           #:log-msg
           ;; Loggign trainer
           #:logging-trainer
           #:log-training-error
           #:log-training-period
           #:log-test-error
           #:log-test-period
           ;; Base trainer
           #:base-trainer
           #:training-counters-and-measurers
           #:prepend-name-to-counters
           ;; Cross entropy softmax classification
           #:softmax-label-chunk*
           #:cesc-trainer
           #:log-dbn-cesc-accuracy
           #:log-dbm-cesc-accuracy
           #:bpn-cesc-error
           #:tack-cross-entropy-softmax-error-on
           ;; Misc
           #:load-weights
           #:save-weights
           ;; BPN setup
           #:arrange-for-renormalizing-activations))

(cl:defpackage :mgl-example-spiral
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd #:mgl-bm #:mgl-bp
        #:mgl-unroll #:mgl-example-util))

(cl:defpackage :mgl-example-mnist
  (:use #:common-lisp #:mgl-util #:mgl-mat #:mgl-train #:mgl-gd #:mgl-cg
        #:mgl-bm #:mgl-bp #:mgl-unroll #:mgl-example-util)
  (:export #:*mnist-dir*
           #:train-mnist))

(cl:defpackage #:mgl-example-movie-review
  (:use #:common-lisp #:mgl-util #:mgl-train #:mgl-gd #:mgl-bm #:mgl-bp
        #:mgl-unroll #:mgl-example-util))

(cl:defpackage :mgl-example-gp
  (:use #:common-lisp #:mgl-util #:mgl-mat #:mgl-train #:mgl-gd #:mgl-gp
        #:mgl-bp #:mgl-example-util))

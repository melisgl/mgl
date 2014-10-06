(cl:defpackage :mgl-example-util
  (:use #:common-lisp #:mgl-util #:mgl-mat #:mgl-core #:mgl-opt #:mgl-gd
        #:mgl-bm #:mgl-bp)
  (:export
   ;; Repeatable experiments
   #:*experiment-random-seed*
   #:run-experiment
   #:with-experiment
   ;; Logging
   #:*example-dir*
   #:time->string
   #:log-msg
   ;; Loggign optimizer
   #:logging-optimizer
   #:log-training-error
   #:log-training-period
   #:log-test-error
   #:log-test-period
   ;; Base optimizer
   #:base-optimizer
   #:training-counters-and-measurers
   #:prepend-name-to-counters
   ;; Cross entropy softmax classification
   #:softmax-label-chunk*
   #:cesc-optimizer
   #:log-dbn-cesc-accuracy
   #:log-dbm-cesc-accuracy
   #:bpn-cesc-error
   ;; Misc
   #:load-weights
   #:save-weights))

(cl:defpackage :mgl-example-mnist
  (:use #:common-lisp #:mgl-util #:mgl-mat #:mgl-core
        #:mgl-opt #:mgl-gd #:mgl-cg
        #:mgl-bm #:mgl-bp #:mgl-unroll #:mgl-example-util)
  (:export #:*mnist-dir*
           #:train-mnist))

(cl:defpackage :mgl-example-gp
  (:use #:common-lisp #:mgl-mat #:mgl-util #:mgl-core #:mgl-opt #:mgl-gd
        #:mgl-gp #:mgl-bp #:mgl-example-util))

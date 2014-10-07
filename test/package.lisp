(cl:defpackage :mgl-test
  (:use #:common-lisp
        #:mgl-mat
        #:mgl-resample
        #:mgl-util
        #:mgl-dataset
        #:mgl-core
        #:mgl-opt
        #:mgl-gd
        #:mgl-cg
        #:mgl-bm
        #:mgl-bp
        #:mgl-unroll
        #:mgl-gp)
  (:export #:test))

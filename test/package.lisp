(cl:defpackage :mgl-test
  (:use #:common-lisp
        #:mgl-resample
        #:mgl-util
        #:mgl-mat
        #:mgl-core
        #:mgl-opt
        #:mgl-gd
        #:mgl-cg
        #:mgl-bm
        #:mgl-bp
        #:mgl-unroll
        #:mgl-gp)
  (:export #:test))

(cl:defpackage :mgl-test
  (:use #:common-lisp
        #:mgl-resample
        #:mgl-util
        #:mgl-mat
        #:mgl-train
        #:mgl-gd
        #:mgl-cg
        #:mgl-bm
        #:mgl-bp
        #:mgl-unroll
        #:mgl-gp)
  (:export #:test))

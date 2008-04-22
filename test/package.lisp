(cl:defpackage :mgl-test
  (:use #:common-lisp #:mgl-util :mgl-train :mgl-gd :mgl-cg :mgl-rbm :mgl-bp
        :mgl-unroll-dbn)
  (:export #:test))

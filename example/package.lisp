(cl:defpackage :mgl-example-util
  (:use #:common-lisp)
  (:export #:*example-dir*
           #:time->string
           #:log-msg
           #:nshuffle-vector
           #:make-random-generator))

(cl:defpackage :mgl-example-spiral
  (:use #:common-lisp #:mgl-util :mgl-train :mgl-gd :mgl-rbm :mgl-bp
        :mgl-unroll-dbn :mgl-example-util))

(cl:defpackage :mgl-example-mnist
  (:use #:common-lisp #:mgl-util :mgl-train :mgl-gd :mgl-rbm :mgl-bp
        :mgl-unroll-dbn :mgl-example-util))

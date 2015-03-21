(cl:defpackage :mgl-example-util
  (:use #:common-lisp #:mgl-mat #:mgl-util #:mgl-log #:mgl-dataset #:mgl-core
        #:mgl-opt #:mgl-gd #:mgl-bm #:mgl-bp)
  (:export
   #:*example-dir*
   #:*example-log-file*
   #:with-example-log))

(cl:defpackage :mgl-example-mnist
  (:use #:common-lisp #:mgl #:mgl-example-util)
  (:export #:*mnist-dir*
           #:train-mnist))

(cl:defpackage :mgl-example-gp
  (:use #:common-lisp #:mgl-mat #:mgl-util #:mgl-log #:mgl-dataset #:mgl-core
        #:mgl-opt #:mgl-gd
        #:mgl-gp #:mgl-bp #:mgl-example-util))

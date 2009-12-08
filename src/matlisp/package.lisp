(cl:defpackage :matlisp
  (:use #:common-lisp)
  (:export #:real-matrix
           #:real-matrix-element-type
           #:real-matrix-store-type
           #:nrows
           #:ncols
           #:number-of-elements
           #:fill-matrix
           #:make-real-matrix-dim
           #:make-real-matrix
           ;;
           #:copy
           #:copy!
           #:gemm!
           #:m+!
           #:scal!
           #:ones
           #:sum
           ;;
           #:*print-matrix*))

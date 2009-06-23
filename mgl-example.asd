;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-example
  :name "Examples for MGL, the machine learning library"
  :author "Gabor Melis"
  :licence "MIT"
  :components ((:module "example"
                :serial t
                :components ((:file "package")
                             (:file "util")
                             (:file "spiral")
                             (:file "mnist")
                             (:file "mnist-dbm")
                             (:file "movie-review"))))
  :depends-on (#:mgl #:cl-ppcre))

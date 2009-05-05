;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-test
  :name "Tests for MGL, the machine learning library"
  :author "Gabor Melis"
  :licence "MIT"
  :components ((:module "test"
                :serial t
                :components ((:file "package")
                             (:file "util")
                             (:file "test-confusion-matrix")
                             (:file "test-util")
                             (:file "test-conjugate-gradient")
                             (:file "test-rbm")
                             (:file "test-backprop")
                             (:file "test-unroll-dbn")
                             (:file "test"))))
  :depends-on (#:mgl))

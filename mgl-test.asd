;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-test
  :name "Tests for MGL, the machine learning library"
  :author "Gabor Melis"
  :licence "MIT"
  :components ((:module "test"
                :serial t
                :components ((:file "package")
                             (:file "util")
                             (:file "test-resample")
                             (:file "test-util")
                             (:file "test-copy")
                             (:file "test-classification")
                             (:file "test-confusion-matrix")
                             (:file "test-conjugate-gradient")
                             (:file "test-boltzmann-machine")
                             (:file "test-backprop")
                             (:file "test-unroll")
                             (:file "test-gaussian-process")
                             (:file "test"))))
  :depends-on (#:mgl #:mgl-mat-test))

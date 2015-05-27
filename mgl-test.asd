;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-test
  :name "Tests for MGL"
  :licence "MIT, see COPYING."
  :author "Gábor Melis"
  :mailto "mega@retes.hu"
  :homepage "http://quotenil.com"
  :description "Test system for MGL, the machine learning library."
  :components ((:module "test"
                :serial t
                :components ((:file "package")
                             (:file "util")
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
  :depends-on (#:mgl))

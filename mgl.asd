;;;; -*- mode: Lisp -*-

;;; See MGL::@MGL-MANUAL for the user guide.
(asdf:defsystem #:mgl
  :licence "MIT, see COPYING."
  :version "0.1.0"
  :name "MGL, the machine learning library"
  :author "Gábor Melis <mega@retes.hu>"
  :mailto "mega@retes.hu"
  :homepage "http://melisgl.github.io/mgl"
  :bug-tracker "https://github.com/melisgl/mgl/issues"
  :source-control (:git "https://github.com/melisgl/mgl.git")
  :description "MGL is a machine learning library for backpropagation
  neural networks, boltzmann machines, gaussian processes and more."
  :components ((:module "src"
                :serial t
                :components ((:file "package")
                             (:file "common")
                             (:file "resample")
                             (:file "util")
                             (:file "log")
                             (:file "dataset")
                             (:file "copy")
                             (:file "core")
                             (:file "feature")
                             (:file "monitor")
                             (:file "counter")
                             (:file "measure")
                             (:file "classification")
                             (:file "optimize")
                             (:file "gradient-descent")
                             (:file "conjugate-gradient")
                             (:file "differentiable-function")
                             (:file "boltzmann-machine")
                             (:file "deep-belief-network")
                             (:file "backprop")
                             (:file "lumps")
                             (:file "unroll")
                             (:file "gaussian-process")
                             (:file "nlp")
                             (:file "mgl")
                             (:file "doc"))))
  :depends-on (#:alexandria #:closer-mop #:array-operations #:lla #:cl-reexport
               #:mgl-gnuplot #:mgl-mat #:mgl-pax #:num-utils #:named-readtables
               #:pythonic-string-reader #:swank)
  :in-order-to ((asdf:test-op (asdf:test-op "mgl/test"))))

(asdf:defsystem #:mgl/test
  :licence "MIT, see COPYING."
  :author "Gábor Melis <mega@retes.hu>"
  :mailto "mega@retes.hu"
  :description "Test system for MGL-GPR."
  :depends-on (#:mgl #:mgl-mat/test)
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
                             (:file "test-rnn")
                             (:file "test-unroll")
                             (:file "test-gaussian-process")
                             (:file "test-nlp")
                             (:file "test"))))
  :perform (asdf:test-op (o s)
             (uiop:symbol-call '#:mgl-test '#:test)))

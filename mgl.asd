;;;; -*- mode: Lisp -*-

(defpackage mgl-system
  (:use #:common-lisp :asdf))

(in-package :mgl-system)

(defsystem #:mgl
  :name "MGL, the machine learning library"
  :author "Gabor Melis"
  :version "0.0.6"
  :licence "MIT"
  :components ((:module "src"
                :serial t
                :components ((:file "package")
                             (:file "util")
                             (:file "copy")
                             (:file "confusion-matrix")
                             (:file "feature")
                             (:file "train")
                             (:file "segment")
                             (:file "classification")
                             (:file "gradient-descent")
                             (:file "conjugate-gradient")
                             (:file "boltzmann-machine")
                             (:file "deep-belief-network")
                             (:file "backprop")
                             (:file "unroll"))))
  :depends-on (:alexandria :closer-mop :array-operations :lla))

(defmethod perform ((o test-op) (c (eql (find-system '#:mgl))))
  (oos 'load-op '#:mgl-test)
  (funcall (intern (symbol-name '#:test)
                   (find-package '#:mgl-test))))

(defmethod operation-done-p ((o test-op) (c (eql (find-system '#:mgl))))
  (values nil))

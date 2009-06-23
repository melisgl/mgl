;;;; -*- mode: Lisp -*-

(defpackage mgl-system
  (:use #:common-lisp :asdf))

(in-package :mgl-system)

(defclass matlisp-source-file (cl-source-file) ())

(defsystem #:mgl
  :name "MGL, the machine learning library"
  :author "Gabor Melis"
  :version "0.0.3"
  :licence "MIT"
  :components ((:module "src"
                :serial t
                :components ((:module "matlisp"
                              :default-component-class matlisp-source-file
                              :serial t
                              :components ((:file "package")
                                           (:file "matrix")
                                           (:file "ref")
                                           (:file "matlisp-impl")))
                             (:file "package")
                             (:file "util")
                             (:file "copy")
                             (:file "confusion-matrix")
                             (:file "feature")
                             (:file "train")
                             (:file "segment")
                             (:file "gradient-descent")
                             (:file "conjugate-gradient")
                             (:file "boltzmann-machine")
                             (:file "deep-belief-network")
                             (:file "backprop")
                             (:file "unroll"))))
  :depends-on (:closer-mop))

(defmethod perform ((o test-op) (c (eql (find-system '#:mgl))))
  (oos 'load-op '#:mgl-test)
  (funcall (intern (symbol-name '#:test)
                   (find-package '#:mgl-test))))

(defmethod operation-done-p ((o test-op) (c (eql (find-system '#:mgl))))
  (values nil))

(defmethod operation-done-p ((o compile-op) (c matlisp-source-file))
  (or (find-package '#:blas)
      (call-next-method)))

(defmethod operation-done-p ((o load-op) (c matlisp-source-file))
  (or (find-package '#:blas)
      (call-next-method)))

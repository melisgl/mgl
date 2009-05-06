;;;; -*- mode: Lisp -*-

(defpackage mgl-system
  (:use #:common-lisp :asdf))

(in-package :mgl-system)

(defclass matlisp-source-file (cl-source-file) ())

(defsystem #:mgl
  :name "MGL, the machine learning library"
  :author "Gabor Melis"
  :version "0.0.2"
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
                             (:file "confusion-matrix")
                             (:file "feature")
                             (:file "train")
                             (:file "segment")
                             (:file "gradient-descent")
                             (:file "conjugate-gradient")
                             (:file "rbm")
                             (:file "dbn")
                             (:file "backprop")
                             (:file "unroll-dbn")))))

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

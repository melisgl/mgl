;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl
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
                             (:file "unroll")
                             (:file "gaussian-process"))))
  :depends-on (#:alexandria #:closer-mop #:array-operations #:lla
                            #:ieee-floats #:mgl-gnuplot))

(defmethod asdf:perform ((o asdf:test-op) (c (eql (asdf:find-system '#:mgl))))
  (asdf:oos 'asdf:load-op '#:mgl-test)
  (funcall (intern (symbol-name '#:test) (find-package '#:mgl-test))))

(defmethod asdf:operation-done-p ((o asdf:test-op)
                                  (c (eql (asdf:find-system '#:mgl))))
  (values nil))

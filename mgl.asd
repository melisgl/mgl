;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl
  :name "MGL, the machine learning library"
  :author "Gabor Melis"
  :version "0.0.1"
  :licence "MIT"
  :components ((:module "src"
                :serial t
                :components ((:file "package")
                             (:file "util")
                             (:file "train")
                             (:file "segment")
                             (:file "gradient-descent")
                             (:file "conjugate-gradient")
                             (:file "rbm")
                             (:file "dbn")
                             (:file "backprop")
                             (:file "unroll-dbn")))))

(defmethod asdf:perform ((o asdf:test-op) (c (eql (asdf:find-system '#:mgl))))
  (asdf:oos 'asdf:load-op '#:mgl-test)
  (funcall (intern (symbol-name '#:test)
                   (find-package '#:mgl-test))))

(defmethod asdf:operation-done-p ((o asdf:test-op)
             
                     (c (eql (asdf:find-system '#:mgl))))
  (values nil))

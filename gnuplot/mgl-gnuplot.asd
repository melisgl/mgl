;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-gnuplot
  :name "MGL Gnuplot"
  :author "Gabor Melis"
  :licence "MIT"
  :components ((:module "src"
                :serial t
                :components ((:file "package")
                             (:file "gnuplot"))))
  :depends-on (#:external-program #:alexandria))

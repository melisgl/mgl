;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-gnuplot
  :name "MGL Gnuplot"
  :licence "MIT, see COPYING."
  :author "Gábor Melis"
  :mailto "mega@retes.hu"
  :homepage "http://quotenil.com"
  :description "Simple Gnuplot interface."
  :components ((:module "src"
                :components ((:module "gnuplot"
                              :serial t
                              :components ((:file "package")
                                           (:file "gnuplot"))))))
  :depends-on (#:external-program #:alexandria))

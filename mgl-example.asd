;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-example
  :name "Examples for MGL"
  :licence "MIT, see COPYING."
  :author "Gábor Melis"
  :mailto "mega@retes.hu"
  :homepage "http://quotenil.com"
  :description "Examples for MGL, the machine learning library."
  :components ((:module "example"
                :serial t
                :components ((:file "package")
                             (:file "util")
                             (:file "spiral")
                             (:file "mnist")
                             (:file "movie-review")
                             (:file "gaussian-process"))))
  :depends-on (#:mgl #:cl-ppcre))

;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-visuals
  :name "Visualization for MGL"
  :licence "MIT, see COPYING."
  :author "Gábor Melis"
  :mailto "mega@retes.hu"
  :homepage "http://quotenil.com"
  :description "Code to generate diagrams from models."
  :components ((:module "src"
                :components ((:module "visuals"
                              :serial t
                              :components ((:file "package")
                                           (:file "dot"))))))
  :depends-on (#:mgl #:cl-ppcre #:cl-dot))

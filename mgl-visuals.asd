;;;; -*- mode: Lisp -*-

(asdf:defsystem #:mgl-visuals
  :name "Visualization for MGL"
  :author "Gabor Melis"
  :licence "MIT"
  :components ((:module "src"
                        :components ((:module "visuals"
                                              :serial t
                                              :components ((:file "package")
                                                           (:file "visuals"))))))
  :depends-on (:mgl :cl-ppcre :cl-dot))

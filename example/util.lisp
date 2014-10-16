(in-package :mgl-example-util)

(defparameter *example-dir*
  (merge-pathnames (make-pathname :directory '(:relative "example"))
                   (make-pathname :name nil :type nil
                                  :defaults (asdf:component-pathname
                                             (asdf:find-system :mgl-example)))))

(defparameter *example-log-file*
  (merge-pathnames "mgl-example.log" *example-dir*))

(defmacro with-example-log (() &body body)
  `(let ((*log-file* *example-log-file*))
     ,@body))

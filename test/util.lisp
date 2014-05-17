(in-package :mgl-test)

(defun ~= (x y &optional (tolerance 0.00001))
  (< (abs (- y x)) tolerance))

(defun log-msg (format &rest args)
  (format *trace-output* "~&")
  (apply #'format *trace-output* format args))

(defmacro do-cuda (() &body body)
  `(loop for enabled in (if (cl-cuda:cuda-available-p)
                            '(nil t)
                            '(nil))
         do (with-cuda (:enabled enabled) ,@body)))

(in-package :mgl-test)

(defun ~= (x y &optional (tolerance 0.00001))
  (< (abs (- y x)) tolerance))

(defun vector= (v1 v2)
  (and (vectorp v1)
       (vectorp v2)
       (every #'= v1 v2)))

(defun log-msg (format &rest args)
  (format *trace-output* "~&")
  (apply #'format *trace-output* format args))

(defmacro do-cuda (() &body body)
  `(loop for enabled in (if (cuda-available-p)
                            '(nil t)
                            '(nil))
         do (with-cuda (:enabled enabled) ,@body)))

(in-package :mgl-test)

(defun test-copy-cons ()
  (let* ((x (cons 1 2))
         (y (copy t (cons 1 2))))
    (assert (equal x y))
    (assert (not (eq x y)))))

(defun test-copy ()
  (test-copy-cons))

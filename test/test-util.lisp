(in-package :mgl-test)

(defun test-break-seq ()
  (assert (equal (mgl-util:break-seq '(2 3) '(0 1 2 3 4 5 6 7 8 9))
                 '((0 1 2 3) (4 5 6 7 8 9))))
  (assert (equal (mgl-util:break-seq '(2 3) '(0))
                 '(() (0)))))

(defun test-stratified-split ()
  (assert (equal (stratified-split '(2 3) '(0 1 2 3 4 5 6 7 8 9) :key #'evenp)
                 '((0 2 1 3) (4 6 8 5 7 9))))
  (assert (equal (stratified-split '(2 3) '(0 1 2 3 4) :key #'evenp)
                 '((0) (2 4 1 3)))))

(defun test-util ()
  (test-break-seq)
  (test-stratified-split)
  (test-confusion-matrix))

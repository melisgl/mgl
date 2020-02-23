(in-package :mgl-test)

(defun test-confusion-matrix-1 ()
  (let ((m (make-instance 'confusion-matrix)))
    (incf (confusion-count m 'cat 'cat) 5)
    (setf (confusion-count m 'cat 'dog) 2)
    (incf (confusion-count m 'dog 'cat) 3)
    (incf (confusion-count m 'dog 'dog) 3)
    (incf (confusion-count m 'dog 'rabbit) 2)
    (incf (confusion-count m 'rabbit 'dog) 1)
    (incf (confusion-count m 'rabbit 'rabbit) 11)
    (pprint m)))

(defun test-confusion-matrix-2 ()
  (let ((m (measure-confusion
            ;; Some compilers may use the same object for all "a"
            ;; strings, so use COPY-SEQ to ensure they are not EQ.
            (mapcar #'copy-seq '("a" "b" "c" "a" "b" "a"))
            (mapcar #'copy-seq '("a" "b" "c" "a" "b" "a"))
            :test 'equal)))
    (assert (equal (multiple-value-list (confusion-matrix-recall m "a"))
                   '(1 3 3)))
    (assert (equal (multiple-value-list (confusion-matrix-precision m "a"))
                   '(1 3 3)))))

(defun test-confusion-matrix ()
  (test-confusion-matrix-1)
  (test-confusion-matrix-2))

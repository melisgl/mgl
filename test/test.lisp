(in-package :mgl-test)

(defun test ()
  (test-confusion-matrix)
  (test-cg)
  (test-rbm)
  (test-bp)
  (test-unroll-dbn)
  (values))

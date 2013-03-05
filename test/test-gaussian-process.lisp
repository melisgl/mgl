(in-package :mgl-test)

(defun test-gp-simple ()
  (let* ((prior (make-instance
                 'prior-gp
                 :mean-fn (constantly 5)
                 :covariance-fn (lambda (x1 x2)
                                  (+ (* 5 (exp (- (expt (/ (- x1 x2) 10) 2))))
                                     1))))
         (posterior (update-gp prior (vector 1 2 3) (vector 2 4 6))))
    (assert (eq prior (update-gp prior (vector) (vector))))
    (gp-means-and-covariances posterior (vector 1.5))
    (let ((posterior2 (update-gp posterior (vector 10 30) (vector 2 4))))
      (multiple-value-bind (means covariances)
          (gp-means-and-covariances posterior2 (vector 1.5))
        (assert (> 0.1 (- 3 (to-scalar
                             (mv-gaussian-random
                              :means means :covariances covariances)))))))))

(defun test-gp ()
  (test-gp-simple))

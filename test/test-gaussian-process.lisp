(in-package :mgl-test)

(defun test-gp-simple ()
  (flet ((v (&rest args)
           (array-to-mat (apply #'vector
                                (mapcar (lambda (x) (flt x)) args))
                         :ctype flt-ctype)))
    (let* ((prior (make-instance
                   'prior-gp
                   :mean-fn (constantly 5)
                   :covariance-fn (lambda (x1 x2)
                                    (+ (* 5 (exp (- (expt (/ (- x1 x2) 10) 2))))
                                       1))))
           (posterior (update-gp prior (v 1 2 3) (v 2 4 6))))
      (assert (eq prior (update-gp prior (v) (v))))
      (gp-means-and-covariances posterior (v 1.5))
      (let ((posterior2 (update-gp posterior (v 10 30) (v 2 4))))
        (multiple-value-bind (means covariances)
            (gp-means-and-covariances posterior2 (v 1.5))
          (assert (> 0.1 (- 3 (mat-as-scalar
                               (mv-gaussian-random
                                :means means :covariances covariances))))))))))

(defun test-gp ()
  (do-cuda ()
    (test-gp-simple)))

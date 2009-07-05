(in-package :mgl-test)

;;; rosenbrock.m This function returns the function value, partial
;;; derivatives and Hessian of the (general dimension) rosenbrock
;;; function, given by:
;;;
;;;   f(x) = sum_{i=1:D-1} 100*(x(i+1) - x(i)^2)^2 + (1-x(i))^2
;;;
;;; where D is the dimension of x. The true minimum is 0 at x = (1 1
;;; ... 1).
;;;
;;; Carl Edward Rasmussen, 2001-07-21.
(defun rosenbrock (x df)
  (declare (type flt-vector x df))
  (let ((d (length x)))
    (assert (= d (length df)))
    ;; df(1:D-1) = - 400*x(1:D-1).*(x(2:D)-x(1:D-1).^2) - 2*(1-x(1:D-1))
    (loop for i below (1- d) do
          (let ((xi (aref x i))
                (xi+1 (aref x (1+ i))))
            (setf (aref df i)
                  (- (* -400 xi (- xi+1 (expt xi 2)))
                     (* 2 (- 1 xi))))))
    ;; df(2:D) = df(2:D) + 200*(x(2:D)-x(1:D-1).^2)
    (setf (aref df (1- d)) (flt 0))
    (loop for i upfrom 1 below d do
          (let ((xi (aref x i))
                (xi-1 (aref x (1- i))))
            (incf (aref df i)
                  (* 200 (- xi (expt xi-1 2))))))
    ;; return function value
    (loop for i below (1- d)
          summing (+ (* 100
                        (expt (- (aref x (1+ i))
                                 (expt (aref x i) 2))
                              2))
                     (expt (- 1 (aref x i)) 2))
          of-type flt)))

(defun test-rosenbrock-cg ()
  (multiple-value-bind (best-weights best-cost)
      (cg #'rosenbrock
          (make-flt-array 100)
          :max-n-line-searches nil
          :max-n-evaluations 2121)
    (assert (~= 0 best-cost))
    (assert (every (lambda (x) (~= 1 x))
                   best-weights))))

(defun square* (x df)
  (declare (type flt-vector x df))
  (assert (= 1 (length x) (length df)))
  (let ((v (aref x 0)))
    (setf (aref df 0) (* 2 v))
    (* v v)))

(defun test-cg-max-evaluations-per-line-search ()
  ;; This is already at the minimum so CG will retturn with 2 failed
  ;; line searches and 0 succesful ones.
  (multiple-value-bind (best-w best-f n-line-searches
                               n-succesful-line-searches n-evaluations)
      (cg #'square*
          (make-flt-array 1)
          :max-n-line-searches 3
          :max-n-evaluations-per-line-search 2)
    (assert (zerop (aref best-w 0)))
    (assert (zerop best-f))
    (assert (= 2 n-line-searches))
    (assert (= 0 n-succesful-line-searches))
    (assert (= n-evaluations 5))))

(defun test-cg ()
  (test-rosenbrock-cg)
  (test-cg-max-evaluations-per-line-search))

(in-package :mgl-gd)

;;;; Online Stochastics Variance Reduced Gradient (OSVRG)
;;;;
;;;; Based on "Accelerating Stochastic Gradient Descent using
;;;; Predictive Variance Reduction" by Rie Johnson and Tong Zhang.
;;;;
;;;; The main idea is to maintain an estimate M of the average (over
;;;; recent examples) gradient for some recent weight W'. Instead of
;;;; the raw gradient G for an example at current weight W, take its
;;;; difference to the gradient at W' for the same example and add
;;;; that to M. So the variance reduced gradient is G - G' + M.
;;;;
;;;; This implementation departs from what's described in the above
;;;; paper by maintaining M and W' as exponential moving averages.

(defclass svrg-trainer (segmented-gd-trainer)
  ((lag :type flt :initarg lag :accessor lag)
   ;; SEGMENT-SET contains all SEGMENTS (inherited from
   ;; SEGMENTED-GD-TRAINER) and determines the layout of
   ;; NORMAL-GRADIENTS, NORMAL-WEIGHTS, LAGGING-WEIGHTS,
   ;; LAGGING-AVERAGE-GRADIENTS.
   (segment-set :accessor segment-set)
   (normal-gradients :type mat :accessor normal-gradients)
   (normal-weights :type mat :accessor normal-weights)
   (lagging-weights :type mat :accessor lagging-weights)
   (lagging-average-gradients :type mat
                              :accessor lagging-average-gradients)))

(define-descriptions (trainer svrg-trainer)
  (lag (lag trainer) "~,5E"))

(defmethod initialize-gradient-sink ((svrg svrg-trainer) source segmentable)
  ;; Let SEGMENTS of SVRG be set up.
  (call-next-method)
  (setf (slot-value svrg 'segment-set)
        (make-instance 'segment-set :segments (segments svrg)))
  (let ((n-weights (segment-set-size (segment-set svrg))))
    (setf (normal-gradients svrg) (make-mat n-weights :ctype flt-ctype))
    (setf (normal-weights svrg) (make-mat n-weights :ctype flt-ctype))
    (setf (lagging-weights svrg) (make-mat n-weights :ctype flt-ctype))
    (setf (lagging-average-gradients svrg)
          (make-mat n-weights :ctype flt-ctype)))
  (segment-set->mat (segment-set svrg) (lagging-weights svrg)))

;;; Save the current weights, install the lagging ones, execute BODY
;;; restore the weights.
(defmacro with-lagging-weights ((svrg) &body body)
  (alexandria:once-only (svrg)
    `(unwind-protect
          (progn
            (segment-set->mat (segment-set ,svrg) (normal-weights ,svrg))
            (segment-set<-mat (segment-set ,svrg) (lagging-weights ,svrg))
            ,@body)
       (segment-set<-mat (segment-set ,svrg) (normal-weights ,svrg)))))

(defmethod train-batch (batch (svrg svrg-trainer) source)
  (let ((n (length batch))
        (random-state (make-random-state *random-state*))
        (curand-state (if (use-cuda-p)
                          (copy-random-state *curand-state*)
                          *curand-state*))
        (lag (lag svrg)))
    (let ((*accumulating-interesting-gradients* t)
          (normal-gradients-sink
            (make-instance 'trivial-sink
                           :segment-set (segment-set svrg)
                           :accumulator (normal-gradients svrg))))
      (fill! (flt 0) (normal-gradients svrg))
      (accumulate-gradients batch (gradient-source svrg)
                            normal-gradients-sink #.(flt 1))
      (add-normal-gradients svrg)
      (update-lagging-weights lag svrg)
      (update-lagging-average-gradient lag n svrg))
    (with-lagging-weights (svrg)
      ;; Sources may not be deterministic (boltzmann machines, dropout
      ;; bpn, etc). In a puny effort to work around that, reinstate
      ;; the random state that was in effect when the normal gradients
      ;; were computed. This breaks down if the random state is used
      ;; by multiple threads (against which we protect in TRAIN) or if
      ;; there are sources of non-determinism other than
      ;; *RANDOM-STATE* and *CURAND-STATE*.
      (let ((*random-state* random-state))
        (with-curand-state (curand-state)
          (accumulate-gradients batch (gradient-source svrg) svrg #.(flt -1)))))
    (add-lagging-average-gradients svrg n)
    (maybe-update-weights svrg n)))

(defun add-segment-set-to-weights (alpha segment-set weights)
  (map-concat (lambda (x y) (axpy! alpha x y))
              segment-set weights :key #'segment-weights))

(defun update-lagging-weights (lag svrg)
  (scal! lag (lagging-weights svrg))
  (add-segment-set-to-weights (- #.(flt 1) lag) (segment-set svrg)
                              (lagging-weights svrg)))

(defun update-lagging-average-gradient (lag n svrg)
  (scal! lag (lagging-average-gradients svrg))
  (axpy! (/ (- #.(flt 1) lag) n) (normal-gradients svrg)
         (lagging-average-gradients svrg)))

(defun map-mat-and-accumulator (fn segment-set mat source sink)
  (do-segment-set (segment :start-in-segment-set start-in-segment-set)
                  segment-set
    (let ((n (segment-size segment)))
      (with-shape-and-displacement (mat n start-in-segment-set)
        (with-sink-accumulator (accumulator (segment source sink))
          (funcall fn mat accumulator))))))

(defun add-normal-gradients (svrg)
  (map-mat-and-accumulator (lambda (normal-gradients accumulator)
                             (axpy! 1 normal-gradients accumulator))
                           (segment-set svrg) (normal-gradients svrg)
                           (gradient-source svrg) svrg))

(defun add-lagging-average-gradients (svrg n)
  (map-mat-and-accumulator (lambda (lagging-average-gradients accumulator)
                             (axpy! n lagging-average-gradients accumulator))
                           (segment-set svrg) (lagging-average-gradients svrg)
                           (gradient-source svrg) svrg))

(defclass trivial-sink (gradient-sink)
  ((segment-set :initarg :segment-set :reader segment-set)
   (accumulator :type flt-vector :initarg :accumulator :reader accumulator)))

(defmethod segments ((sink trivial-sink))
  (segments (segment-set sink)))

(defmethod map-gradient-sink (fn (sink trivial-sink))
  (let ((segment-set (segment-set sink))
        (accumulator (accumulator sink)))
    (do-segment-set (segment :start-in-segment-set start) segment-set
      (funcall fn segment accumulator start))))

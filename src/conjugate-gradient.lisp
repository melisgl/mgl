;;;; Conjugate gradient descent along the lines described by Carl
;;;; Rasmussen at
;;;; http://www.cs.toronto.edu/~delve/methods/mlp-ese-1/minimize.ps.gz
;;;; and http://www.kyb.tuebingen.mpg.de/bs/people/carl/code/minimize/ .
;;;;
;;;; Licensing needs to be clarified with upstream author.

(in-package :mgl-cg)

(defvar *default-int* (flt 0.1)
  "Don't reevaluate within INT of the limit of the current bracket.")
(defvar *default-ext* (flt 3)
  "Extrapolate maximum EXT times the current step-size.")
(defvar *default-sig* (flt 0.1)
  "SIG and RHO are the constants controlling the Wolfe-Powell
conditions. SIG is the maximum allowed absolute ratio between previous
and new slopes (derivatives in the search direction), thus setting SIG
to low (positive) values forces higher precision in the
line-searches.")
(defvar *default-rho* (flt 0.05)
  "RHO is the minimum allowed fraction of the expected (from the slope
at the initial point in the linesearch). Constants must satisfy 0 <
RHO < SIG < 1.")
(defvar *default-ratio* (flt 10)
  "Maximum allowed slope ratio.")
(defvar *default-max-n-line-searches* nil)
(defvar *default-max-n-evaluations-per-line-search* 20)
(defvar *default-max-n-evaluations* nil)

(defun negate-vector (v &key result)
  (declare (type flt-vector v))
  (unless result
    (setq result (make-flt-array (length v))))
  (map-into result #'- v))

(defun inner* (v1 v2)
  (declare (type flt-vector v1 v2))
  (assert (= (length v1) (length v2)))
  (let ((sum #.(flt 0)))
    (loop for i below (length v1)
          do (incf sum (* (aref v1 i) (aref v2 i))))
    sum))

(defun v1=v2+c*v3 (v1 v2 c v3)
  (declare (type flt-vector v1 v2 v3)
           (type flt c))
  (assert (= (length v1) (length v2) (length v3)))
  (dotimes (i (length v1))
    (setf (aref v1 i)
          (+ (aref v2 i)
             (* c (aref v3 i))))))

(defun polack-ribiere (old-df new-df)
  (/ (- (inner* new-df new-df)
        (inner* old-df new-df))
     (inner* old-df old-df)))

(defun update-direction (s old-df new-df)
  (let ((c (polack-ribiere old-df new-df)))
    (dotimes (i (length s))
      (setf (aref s i)
            (- (* c (aref s i))
               (aref new-df i))))))

(deftype limit () `(or null (integer 1)))

(defun check-limit (value limit)
  (or (null limit)
      (< value limit)))

;;; This unholy mess is translated from matlab code.
(defun cg (fn w &key (max-n-line-searches *default-max-n-line-searches*)
           (max-n-evaluations-per-line-search
            *default-max-n-evaluations-per-line-search*)
           (max-n-evaluations *default-max-n-evaluations*)
           (sig *default-sig*) (rho *default-rho*)
           (int *default-int*) (ext *default-ext*)
           (ratio *default-ratio*)
           spare-vectors)
  "Minimize a differentiable multivariate function with conjugate
gradient. The Polack-Ribiere flavour of conjugate gradients is used to
compute search directions, and a line search using quadratic and cubic
polynomial approximations and the Wolfe-Powell stopping criteria is
used together with the slope ratio method for guessing initial step
sizes. Additionally a bunch of checks are made to make sure that
exploration is taking place and that extrapolation will not be
unboundedly large.

FN is a function of two parameters: WEIGHTS and DERIVATIVES.
WEIGHTS is an FLT-VECTOR of the same size as W that is where the
search start from. DERIVATIVES is also and FLT-VECTOR of that size and
it is where FN shall place the partial derivatives. FN returns the
value (of type FLT) of the function that is being minimized.

CG performs a number of line searches and invokes FN at each step. A
line search invokes FN at most MAX-N-EVALUATIONS-PER-LINE-SEARCH
number of times and can succeed in improving the minimum by the
sufficient margin or it can fail. Note, the even a failed line search
may improve further and hence change the weights it's just that the
improvement was deemed too small. CG stops when either:

- two line searches fail in a row
- MAX-N-LINE-SEARCHES is reached
- MAX-N-EVALUATIONS is reached

CG returns an FLT-VECTOR that contains the best weights, the minimum
\(of type FLT), the number of line searches performed, the number of
succesful line searches and the number of evaluations.

When using MAX-N-EVALUATIONS remember that there is an extra
evaluation of FN before the first line search.

SPARE-VECTORS is a list of preallocated FLT-VECTORs of the same size
as W. Passing 6 of them covers the current need of the algorithm and
it will not cons up vectors of size W at all.

NOTE: If the function terminates within a few iterations, it could be
an indication that the function values and derivatives are not
consistent \(ie, there may be a bug in the implementation of FN
function).

SIG and RHO are the constants controlling the Wolfe-Powell conditions.
SIG is the maximum allowed absolute ratio between previous and new
slopes (derivatives in the search direction), thus setting SIG to low
\(positive) values forces higher precision in the line-searches. RHO is
the minimum allowed fraction of the expected (from the slope at the
initial point in the linesearch). Constants must satisfy 0 < RHO < SIG
< 1. Tuning of SIG (depending on the nature of the function to be
optimized) may speed up the minimization; it is probably not worth
playing much with RHO."
  ;; The code falls naturally into 3 parts, after the initial line
  ;; search is started in the direction of steepest descent. 1) we
  ;; first enter a while loop which uses point 1 (p1) and (p2) to
  ;; compute an extrapolation (p3), until we have extrapolated far
  ;; enough (Wolfe-Powell conditions). 2) if necessary, we enter the
  ;; second loop which takes p2, p3 and p4 chooses the subinterval
  ;; containing a (local) minimum, and interpolates it, unil an
  ;; acceptable point is found (Wolfe-Powell conditions). Note, that
  ;; points are always maintained in order p0 <= p1 <= p2 < p3 < p4.
  ;; 3) compute a new search direction using conjugate gradients
  ;; (Polack-Ribiere flavour), or revert to steepest if there was a
  ;; problem in the previous line-search.
  (declare (type flt-vector w)
           (type flt sig rho int ext ratio))
  (assert (< 0 rho sig 1))
  (check-type max-n-line-searches limit)
  (check-type max-n-evaluations-per-line-search (integer 1))
  (check-type max-n-evaluations limit)
  (assert (every (lambda (v)
                   (and (typep v 'flt-vector)
                        (= (length w) (length v))))
                 spare-vectors))
  (flet ((get-w-sized-vector ()
           (let ((v (pop spare-vectors)))
             (declare (type (or null flt-vector) v))
             (if v
                 (fill v #.(flt 0))
                 (make-flt-array (length w))))))
    (without-float-traps
      (let* ((df0 (get-w-sized-vector))
             (df3 (get-w-sized-vector))
             (w3 (get-w-sized-vector))
             (f0 (funcall fn w df0))
             ;; direction
             (s (negate-vector df0 :result (get-w-sized-vector)))
             (d0 (- (inner* s s)))
             (x3 (/ (- 1 d0)))
             f1 f2 f3 f4
             ;; slopes
             d1 d2 d3 d4
             ;; steps
             x1 x2 x4
             (ls-failed nil)
             (best-w (get-w-sized-vector))
             (best-df (get-w-sized-vector))
             (best-f f0)
             (n-line-searches 0)
             (n-succesful-line-searches 0)
             (n-evaluations 1))
        (declare (type flt-vector df0 df3 w3 s best-w best-df)
                 (type flt f0 d0 best-f))
        (while (and (check-limit n-line-searches max-n-line-searches)
                    (check-limit n-evaluations max-n-evaluations))
          (incf n-line-searches)
          (replace best-w w)
          (replace best-df df0)
          (setq best-f f0)
          (let ((n-evaluations-per-line-search 0))
            (flet ((update3 ()
                     (incf n-evaluations)
                     (incf n-evaluations-per-line-search)
                     (v1=v2+c*v3 w3 w x3 s)
                     (setq f3 (funcall fn w3 df3))
                     (when (< f3 best-f)
                       (replace best-w w3)
                       (replace best-df df3)
                       (setq best-f f3)))
                   (check-evaluation-limits ()
                     (and (check-limit n-evaluations-per-line-search
                                       max-n-evaluations-per-line-search)
                          (check-limit n-evaluations max-n-evaluations))))
              ;; extrapolating
              (while t
                (setq x2 0 f2 f0 d2 d0 f3 f0)
                (replace df3 df0)
                (update3)
                (setq d3 (inner* df3 s))
                ;; are we done extrapolating?
                (when (or (> d3 (* sig d0))
                          (> f3 (+ f0 (* x3 rho d0)))
                          (not (check-evaluation-limits)))
                  (return))
                (setq x1 x2 f1 f2 d1 d2
                      x2 x3 f2 f3 d2 d3)
                ;; cubic extrapolation
                (let ((a (+ (* 6 (- f1 f2))
                            (* 3 (+ d2 d1) (- x2 x1))))
                      (b (- (* 3 (- f2 f1))
                            (* (+ (* 2 d1) d2)
                               (- x2 x1)))))
                  (setq x3 (- x1 (/ (* d1 (expt (- x2 x1) 2))
                                    (+ b (sqrt (- (expt b 2)
                                                  (* a d1 (- x2 x1)))))))))
                (cond
                  ;; numerical problems?
                  ((or (not (realp x3))
                       (float-nan-p x3)
                       (float-infinity-p x3)
                       (minusp x3)
                       ;; or beyond extrapolation point?
                       (> x3 (* x2 ext)))
                   (setq x3 (* x2 ext)))
                  ((< x3 (+ x2 (* int (- x2 x1))))
                   (setq x3 (+ x2 (* int (- x2 x1)))))))
              ;; interpolation
              (while (and (or (> (abs d3) (- (* sig d0)))
                              (> f3 (+ f0 (* x3 rho d0))))
                          (check-evaluation-limits))
                ;; choose subinterval
                (if (or (plusp d3) (> f3 (+ f0 (* x3 rho d0))))
                    (setq x4 x3 f4 f3 d4 d3)
                    (setq x2 x3 f2 f3 d2 d3))
                (if (> f4 f0)
                    ;; quadratic interpolation
                    (setq x3 (- x2 (/ (* 0.5 d2 (expt (- x4 x2) 2))
                                      (- f4 f2 (* d2 (- x4 x2))))))
                    ;; cubic interpolation
                    (let ((a (+ (/ (* 6 (- f2 f4))
                                   (- x4 x2))
                                (* 3 (+ d4 d2))))
                          (b (- (* 3 (- f4 f2))
                                (* (+ (* 2 d2) d4)
                                   (- x4 x2)))))
                      (setq x3 (+ x2
                                  (/ (- (sqrt (- (* b b)
                                                 (* a d2
                                                    (expt (- x4 x2) 2))))
                                        b)
                                     a)))))
                ;; bisect on numerical problem
                (when (or (float-nan-p x3) (float-infinity-p x3))
                  (setq x3 (/ (+ x2 x4) 2)))
                ;; don't accept too close
                (setq x3 (max (min x3 (- x4 (* int (- x4 x2))))
                              (+ x2 (* int (- x4 x2)))))
                (update3)
                (setq d3 (inner* df3 s)))
              (cond ((and (< (abs d3) (- (* sig d0)))
                          (< f3 (+ f0 (* x3 rho d0))))
                     (incf n-succesful-line-searches)
                     (v1=v2+c*v3 w w x3 s)
                     (setq f0 f3)
                     (update-direction s df0 df3)
                     (rotatef df0 df3)
                     (setq d3 d0
                           d0 (inner* df0 s))
                     (when (plusp d0)
                       (negate-vector df0 :result s)
                       (setq d0 (- (inner* s s))))
                     (setq x3 (* x3
                                 (min ratio
                                      (/ d3
                                         (- d0
                                            least-positive-double-float))))
                           ls-failed nil))
                    (t
                     ;; restore best point so far
                     (replace w best-w)
                     (replace df0 best-df)
                     (setq f0 best-f)
                     ;; line search failed twice in a row ?
                     (when (or ls-failed
                               (not (check-limit n-line-searches
                                                 max-n-line-searches)))
                       (return))
                     ;; try steepest descent
                     (negate-vector df0 :result s)
                     (setq d0 (- (inner* s s))
                           x3 (/ (- 1 d0))
                           ls-failed t))))))
        (values best-w best-f
                n-line-searches n-succesful-line-searches n-evaluations)))))


;;;; Trainer

(defclass cg-trainer ()
  ((batch-size
    :initarg :batch-size :accessor batch-size
    :documentation "After having gone through BATCH-SIZE number of
inputs weights are updated.")
   (n-inputs :initform 0 :initarg :n-inputs :accessor n-inputs)
   (cg-args :initform '() :initarg :cg-args :accessor cg-args)
   (segment-filter
    :initform (constantly t)
    :initarg :segment-filter :reader segment-filter
    :documentation "A predicate function on segments that filters out
uninteresting segments. Called from INITIALIZE-TRAINER.")
   (segment-set :reader segment-set :documentation "Segments to train.")
   (weights :initform nil :accessor weights :type (or null flt-vector))
   (spare-vectors
    :initform nil :accessor spare-vectors :type list
    :documentation "Pre-allocated vectors to make CG less consy.")
   (accumulator1
    :initform nil :reader accumulator1
    :documentation "This is where COMPUTE-BATCH-COST-AND-DERIVE should
leave the derivatives."))
  (:documentation "Updates all weights simultaneously after chewing
through BATCH-SIZE inputs."))

(defmethod n-inputs-until-update ((trainer cg-trainer))
  ;; deal with varying batch size gracefully
  (- (batch-size trainer)
     (mod (n-inputs trainer) (batch-size trainer))))

(defmethod map-segment-gradient-accumulators (fn (trainer cg-trainer))
  (let ((segment-set (segment-set trainer))
        (accumulator1 (accumulator1 trainer)))
    (do-segment-set (segment :start-in-segment-set start) segment-set
      (funcall fn segment start accumulator1))))

(defmethod segments ((trainer cg-trainer))
  (segments (segment-set trainer)))

(defgeneric compute-batch-cost-and-derive (batch trainer learner)
  (:documentation "Return the total cost (that LEARNER is trying to
minimize) of all samples in BATCH and add the derivatives to
ACCUMULATOR1 of TRAINER."))

(defmethod initialize-trainer ((trainer cg-trainer) segmentable)
  (setf (slot-value trainer 'segment-set)
        (make-instance 'segment-set
                       :segments (remove-if-not (segment-filter trainer)
                                                (list-segments segmentable))))
  (let ((n (segment-set-size (segment-set trainer))))
    (setf (weights trainer) (make-flt-array n)
          (spare-vectors trainer) (loop repeat 6
                                        collect (make-flt-array n)))))

(defun process-batch (trainer learner batch weights derivatives)
  (let ((segment-set (segment-set trainer)))
    (segment-set<-weights segment-set weights)
    (setf (slot-value trainer 'accumulator1) derivatives)
    (compute-batch-cost-and-derive batch trainer learner)))

(defmethod train (sampler (trainer cg-trainer) learner)
  (while (not (finishedp sampler))
    (let ((samples (sample-batch sampler (batch-size trainer))))
      (train-batch samples trainer learner))))

(defmethod train-batch (samples (trainer cg-trainer) learner)
  (let ((weights (weights trainer)))
    (cond ((= (length samples) (batch-size trainer))
           (segment-set->weights (segment-set trainer) weights)
           (multiple-value-prog1
               (apply #'cg (lambda (weights derivatives)
                             (process-batch trainer learner
                                            samples weights
                                            derivatives))
                      weights
                      :spare-vectors (spare-vectors trainer)
                      (cg-args trainer))
             (segment-set<-weights (segment-set trainer) (weights trainer))
             (incf (n-inputs trainer) (length samples))))
          (t
           ;; Updating on shorter than prescribed batches is
           ;; dangerous.
           (warn "MGL-CG: only ~S samples in batch of size ~S. Skipping it."
                 (length samples) (batch-size trainer))))))


;;;; Decay

;;; Weight decay could be explicitly implemented in the learners, for
;;; instance in the case of a BPN by adding some lumps to include
;;; weight decay in the cost and get the derivatives for free but
;;; that's kind of wasteful as the weights and consequently the decay
;;; does not change in the batch.
(defclass decayed-cg-trainer-mixin ()
  ((segment-decay-fn
    :initform nil :initarg :segment-decay-fn :accessor segment-decay-fn
    :documentation "If not NIL it's a designator for a function that
returns a decay of type FLT for a given segment. For convenience NIL
is also treated as 0 decay."))
  (:documentation "Mix this before a CG based trainer to conveniently
add decay on a per-segment basis."))

(defmethod mgl-cg:compute-batch-cost-and-derive
    (batch (trainer decayed-cg-trainer-mixin) learner)
  (let* ((cost (flt (call-next-method)))
         (segment-decay-fn (segment-decay-fn trainer))
         (accumulator1 (accumulator1 trainer)))
    (declare (type flt-vector accumulator1)
             (type flt cost))
    (do-segment-set (segment :start-in-segment-set segment-start)
        (segment-set trainer)
      (let ((decay (funcall segment-decay-fn segment)))
        (declare (type (or null flt) decay))
        (when (and decay (not (zerop decay)))
          ;; Because d(regularizer*x^2)/dx = 2*penalty*x hence
          ;; regularizer=decay/2.
          (let* ((decay (* (length batch) decay))
                 (regularizer (/ decay 2)))
            (with-segment-weights ((weights start end) segment)
              (declare (optimize (speed 3)))
              (loop for i upfrom start below end
                    for j upfrom segment-start
                    do (let ((x (aref weights i)))
                         (incf cost (* regularizer x x))
                         (incf (aref accumulator1 j) (* decay x)))))))))
    cost))

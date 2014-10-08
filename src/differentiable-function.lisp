(in-package :mgl-diffun)

(defsection @mgl-diffun (:title "Differentiable Functions")
  (diffun class)
  (fn (reader diffun))
  (parameter-indices (reader diffun))
  (weight-indices (reader diffun)))

(defclass diffun ()
  ((fn
    :initarg :fn :reader fn
    :documentation "A real valued lisp function. It may have any
    number of parameters.")
   (parameter-indices
    :initform ()
    :initarg :parameter-indices
    :reader parameter-indices
    :documentation "The list of indices of parameters that we don't
    optimize. Values for these will come from the DATASET argument of
    MINIMIZE.")
   (weight-indices
    :initform ()
    :initarg :weight-indices
    :reader weight-indices
    :documentation "The list of indices of parameters to be optimized,
    the values of which will come from the WEIGHTS argument of
    MINIMIZE."))
  (:documentation "DIFFUN dresses a lisp function (in its FN slot) as
  a gradient source (see @MGL-OPT-GRADIENT-SOURCE) which allows it to
  be used in MINIMIZE. See the examples in MGL-GD:@MGL-GD and
  MGL-CG:@MGL-CG."))

;;; Build an argument list for FN, the lisp function, of the DIFF-FN
;;; DIFFUN from WEIGHTS and ARGS based on
;;; PARAMETER-INDICES and WEIGHT-INDICES.
(defun merge-weights-and-arguments (diff-fn weights args)
  (let ((all-args (make-list (+ (length (weight-indices diff-fn))
                                (length (parameter-indices diff-fn))))))
    (loop for arg in args
          for parameter-index in (parameter-indices diff-fn)
          do (setf (elt all-args parameter-index) arg))
    (loop for weight across (mat-to-array weights)
          for weight-index in (weight-indices diff-fn)
          do (setf (elt all-args weight-index) weight))
    all-args))

;;; Call FN of DIFF-FN with WEIGHTS and ARGS.
(defun evaluate-diffun (diff-fn weights args)
  (apply (fn diff-fn) (merge-weights-and-arguments diff-fn weights args)))

;;; This is the main extension point for gradient sources.
(defmethod accumulate-gradients* ((diff-fn diffun) gradient-sink
                                  batch multiplier valuep)
  (let ((cost 0))
    (do-gradient-sink ((weights accumulator) gradient-sink)
      (loop for args in batch
            do (add-diffun-gradients
                diff-fn weights args accumulator multiplier)
               (when valuep
                 (incf cost (evaluate-diffun diff-fn
                                                              weights args)))))
    (when valuep
      cost)))

;;; Differentiate DIFF-FN at the point determined by WEIGHTS and ARGS
;;; by WEIGHTS and and the results times MULTIPLIER to ACCUMULATOR.
(defun add-diffun-gradients (diff-fn weights args accumulator multiplier)
  (let ((args (merge-weights-and-arguments diff-fn weights args)))
    (loop for weight-index in (weight-indices diff-fn)
          for i upfrom 0
          do (incf (mref accumulator i)
                   (* multiplier
                      (differentiate-numerically
                       (fn diff-fn) args weight-index))))))

;;; Approximate first derivative of a function by (f(x+d)-f(x))/d. FN
;;; is an arbitrary real valued lisp function. ARGS is a valid
;;; argument list for FN and INDEX is the index of the parameter by
;;; which we want to differentiate.
(defun differentiate-numerically (fn args index &key (delta 0.0001))
  (/ (- (apply fn (let ((args (copy-seq args)))
                    (setf (elt args index) (+ (elt args index) delta))
                    args))
        (apply fn args))
     delta))

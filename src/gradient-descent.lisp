(in-package :mgl-gd)

(in-readtable pythonic-string-syntax)

(defsection @mgl-gd (:title "Gradient Descent")
  "Gradient descent is a first-order optimization algorithm. Relying
  completely on first derivatives, it does not even evaluate the
  function to be minimized. Let's see how to minimize a numerical lisp
  function with respect to some of its parameters."
  (sgd.lisp
   (include #.(asdf:system-relative-pathname :mgl "example/sgd.lisp")
            :header-nl "```commonlisp" :footer-nl "```"))
  "We are going to see a number of accessors for optimizer paramaters.
  In general, it's allowed to SETF real slot accessors (as opposed to
  readers and writers) at any time during optimization and so is
  defining a method on an optimizer subclass that computes the value
  in any way. For example, to decay the learning rate on a per
  mini-batch basis:

  ```commonlisp
  (defmethod learning-rate ((optimizer my-sgd-optimizer))
    (* (slot-value optimizer 'learning-rate)
       (expt 0.998
             (/ (n-instances optimizer) 60000))))
  ```"
  (@mgl-gd-batch-gd-optimizer section)
  (@mgl-gd-segmented-gd-optimizer section)
  (@mgl-gd-per-weight-optimization section)
  (@mgl-gd-utilities section))

;;;; The class hierarchy:
;;;;
;;;; ITERATIVE-OPTIMIZER
;;;;   BASE-GD-OPTIMIZER [not exported]
;;;;     GD-OPTIMIZER [not exported]
;;;;       BATCH-GD-OPTIMIZER
;;;;         SGD-OPTIMIZER
;;;;         ADAM-OPTIMIZER
;;;;         NORMALIZED-BATCH-GD-OPTIMIZER
;;;;       PER-WEIGHT-BATCH-GD-OPTIMIZER
;;;;     SEGMENTED-GD-OPTIMIZER
;;;;   MGL-CG:CG-OPTIMIZER

;;; Abstract gradient descent optimizer base class
(defclass base-gd-optimizer (iterative-optimizer)
  ())

(defmethod minimize* ((optimizer base-gd-optimizer) gradient-source
                      weights dataset)
  (let ((sampler (if (typep dataset 'sequence)
                     (make-random-sampler dataset)
                     dataset)))
    (while (and (not (terminate-optimization-p (n-instances optimizer)
                                               (termination optimizer)))
                (not (finishedp sampler)))
      (let ((batch (list-samples sampler (n-instances-until-update optimizer))))
        (accumulate-gradients* gradient-source optimizer batch 1 nil)
        (maybe-update-weights optimizer gradient-source (length batch))))))

(defgeneric n-instances-until-update (optimizer)
  (:documentation "Return the largest number of inputs guaranteed not
  to cause a change in the learner being trained."))

(defgeneric maybe-update-weights (optimizer gradient-source n-new-inputs)
  (:documentation "Update the weights being trained. N-NEW-INPUTS have
  been seen since the last time this was called."))


(defsection @mgl-gd-batch-gd-optimizer (:title "Batch Based Optimizers")
  "First let's see everything common to all batch based optimizers,
  then discuss @MGL-GD-SGD-OPTIMIZER, @MGL-GD-ADAM-OPTIMIZER and
  @MGL-GD-NORMALIZED-BATCH-GD-OPTIMIZER. All batch based optimizers
  are `ITERATIVE-OPTIMIZER`s, so see @MGL-OPT-ITERATIVE-OPTIMIZER
  too."
  (batch-gd-optimizer class)
  (batch-size (accessor gd-optimizer))
  (learning-rate (accessor gd-optimizer))
  (momentum (accessor gd-optimizer))
  (momentum-type (reader gd-optimizer))
  (weight-decay (accessor gd-optimizer))
  (weight-penalty (accessor gd-optimizer))
  (use-segment-derivatives-p (reader gd-optimizer))
  (after-update-hook (accessor gd-optimizer))
  (before-update-hook (accessor batch-gd-optimizer))
  (@mgl-gd-sgd-optimizer section)
  (@mgl-gd-adam-optimizer section)
  (@mgl-gd-normalized-batch-gd-optimizer section))

;;; Common base class for BATCH-GD-OPTIMIZER, and
;;; PER-WEIGHT-BATCH-GD-OPTIMIZER.
(defclass gd-optimizer (base-gd-optimizer)
  ((segment-set
    :reader segment-set
    :documentation "The set of segments that are to be trained. The
    ACCUMULATOR, WEIGHT-DELTAS, etc vectors are indexed by SEGMENT-SET
    indices.")
   ;; A MAT into which the gradients are summed. Not allocated if
   ;; USE-SEGMENT-DERIVATIVES-P.
   (accumulator :type mat :accessor accumulator)
   (use-segment-derivatives-p
    :initform nil
    :initarg :use-segment-derivatives-p
    :reader use-segment-derivatives-p
    :documentation "Save memory if both the gradient source (the model
    being optimized) and the optimizer support this feature. It works
    like this: the accumulator into which the gradient source is asked
    to place the derivatives of a segment will be SEGMENT-DERIVATIVES
    of the segment. This allows the optimizer not to allocate an
    accumulator matrix into which the derivatives are summed.")
   (weight-deltas :type mat :accessor weight-deltas)
   (batch-size
    :initform 1
    :initarg :batch-size :accessor batch-size
    :documentation "After having gone through BATCH-SIZE number of
    inputs, weights are updated. With BATCH-SIZE 1, one gets
    Stochastics Gradient Descent. With BATCH-SIZE equal to the number
    of instances in the dataset, one gets standard, 'batch' gradient
    descent. With BATCH-SIZE between these two extremes, one gets the
    most practical 'mini-batch' compromise.")
   (learning-rate
    :initform 0.1 :initarg :learning-rate :accessor learning-rate
    :documentation "This is the step size along the gradient. Decrease
    it if optimization diverges, increase it if it doesn't make
    progress.")
   (momentum
    :initform 0 :initarg :momentum :accessor momentum
    :documentation "A value in the [0, 1) interval. MOMENTUM times the
    previous weight change is added to the gradient. 0 means no
    momentum.")
   (momentum-type
    :initform :normal :initarg :momentum-type :reader momentum-type
    :type (member :none :normal :nesterov)
    :documentation "One of :NORMAL, :NESTEROV or :NONE. For pure
    optimization Nesterov's momentum may be better, but it may also
    increases chances of overfitting. Using :NONE is equivalent to 0
    momentum, but it also uses less memory. Note that with :NONE,
    MOMENTUM is ignored even it it is non-zero.")
   (weight-decay
    :initform 0 :initarg :weight-decay :accessor weight-decay
    :documentation "An L2 penalty. It discourages large weights, much
    like a zero mean gaussian prior. WEIGHT-DECAY * WEIGHT is added to
    the gradient to penalize large weights. It's as if the function
    whose minimum is sought had WEIGHT-DECAY*sum_i{0.5 * WEIGHT_i^2}
    added to it.")
   (weight-penalty
    :initform 0 :initarg :weight-penalty :accessor weight-penalty
    :documentation "An L1 penalty. It encourages sparsity.
    SIGN(WEIGHT) * WEIGHT-PENALTY is added to the gradient pushing the
    weight towards negative infinity. It's as if the function whose
    minima is sought had WEIGHT-PENALTY*sum_i{abs(WEIGHT_i)} added to
    it. Putting it on feature biases consitutes a sparsity constraint
    on the features.")
   (after-update-hook
    :type list
    :initform () :initarg :after-update-hook :accessor after-update-hook
    :documentation "A list of functions with no arguments called after
    each weight update."))
  (:documentation "Gradient descent optimizer with momentum, weight
  decay, weight penalty. Batch size and all other parameters can be
  changed during training. One may even want to subclass this
  optimizer, define a method for BATCH-SIZE make it a function of
  N-INSTANCES.

  Depending on BATCH-SIZE, this may be stochastic or non-stochastic
  gradient descent."))

(defmethod print-object ((optimizer gd-optimizer) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (optimizer stream :type t :identity t)
      (format stream "~S" (ignore-errors (segment-set optimizer)))))
  optimizer)

(define-descriptions (optimizer gd-optimizer)
  n-instances segment-set
  (learning-rate (learning-rate optimizer) "~,5E")
  (momentum (momentum optimizer) "~,5E")
  momentum-type
  (weight-decay (weight-decay optimizer) "~,5E")
  (weight-penalty (weight-penalty optimizer) "~,5E")
  (n-after-upate-hook (length (after-update-hook optimizer)) "~S")
  batch-size)

(defmethod initialize-optimizer* ((optimizer gd-optimizer) source weights
                                  dataset)
  (when (next-method-p)
    (call-next-method))
  (setf (slot-value optimizer 'segment-set)
        (make-instance 'segment-set :segments weights))
  (let ((n-weights (size (segment-set optimizer))))
    (unless (use-segment-derivatives-p optimizer)
      (setf (accumulator optimizer) (make-mat n-weights)))
    (unless (eq (momentum-type optimizer) :none)
      (setf (weight-deltas optimizer) (make-mat n-weights)))))

(defmethod segments ((optimizer gd-optimizer))
  (segments (segment-set optimizer)))

(defmethod map-gradient-sink (fn (optimizer gd-optimizer))
  (let ((segment-set (segment-set optimizer))
        (use-segment-derivatives-p (use-segment-derivatives-p optimizer)))
    (if use-segment-derivatives-p
        (do-segment-set (segment start) segment-set
          (declare (ignore start))
          (funcall fn segment (segment-derivatives segment)))
        (let ((accumulator (accumulator optimizer)))
          (do-segment-set (segment start) segment-set
            (with-shape-and-displacement
                (accumulator (mat-size (segment-weights segment)) start)
              (funcall fn segment accumulator)))))))

(defclass batch-gd-optimizer (gd-optimizer)
  ((n-instances-in-batch
    :initform 0 :initarg :n-instances-in-batch :accessor n-instances-in-batch)
   (before-update-hook
    :type list :initform () :initarg :before-update-hook
    :accessor before-update-hook
    :documentation "A list of functions of no parameters. Each
    function is called just before a weight update takes place (after
    accumulated gradients have been divided the length of the batch).
    Convenient to hang some additional gradient accumulating code
    on."))
  (:documentation "Another abstract base class for gradient based
  optimizers tath updates all weights simultaneously after chewing
  through BATCH-SIZE inputs. See subclasses SGD-OPTIMIZER,
  ADAM-OPTIMIZER and NORMALIZED-BATCH-GD-OPTIMIZER.

  PER-WEIGHT-BATCH-GD-OPTIMIZER may be a better choice when some
  weights can go unused for instance due to missing input values."))

(define-descriptions (optimizer batch-gd-optimizer :inheritp t)
  (n-before-upate-hook (length (before-update-hook optimizer)) "~S"))

(defmethod n-instances-until-update ((optimizer batch-gd-optimizer))
  ;; BATCH-SIZE may be setf'ed to a value lower than N-INSTANCES-IN-BATCH
  (max 0 (- (batch-size optimizer)
            (n-instances-in-batch optimizer))))


(defsection @mgl-gd-sgd-optimizer (:title "SGD Optimizer")
  (sgd-optimizer class))

(defclass sgd-optimizer (batch-gd-optimizer)
  ()
  (:documentation """With BATCH-SIZE 1 this is Stochastic Gradient
  Descent. With higher batch sizes, one gets mini-batch and Batch
  Gradient Descent.

  Assuming that ACCUMULATOR has the sum of gradients for a mini-batch,
  the weight update looks like this:

  $$\Delta_w^{t+1} = momentum \Delta_w^t
    + \frac{accumulator}{batchsize}
    + l_2 w + l_1 sign(w)$$

  $$w^{t+1} = w^{t} - learningrate \Delta_w$$

  which is the same as the more traditional formulation:

  $$\Delta_w^{t+1} = momentum * \Delta_w^{t}
    + learningrate \left(\frac{\frac{df}{dw}}{batchsize}
                         + l_2 w + l_1 sign(w)\right)$$

  $$w^{t+1} = w^{t} - \Delta_w$$

  but the former works better when batch size, momentum or learning
  rate change during the course of optimization. The above is with
  normal momentum, Nesterov's momentum (see MOMENTUM-TYPE) momentum is
  also available.

  See @MGL-GD-BATCH-GD-OPTIMIZER for the description of the various
  options common to all batch based optimizers."""))

(defmethod maybe-update-weights ((optimizer sgd-optimizer)
                                 gradient-source n-new-inputs)
  (when (<= (batch-size optimizer)
            (incf (n-instances-in-batch optimizer) n-new-inputs))
    (ecase (momentum-type optimizer)
      ((:none)
       (update-all-weights/sgd-no-momentum optimizer))
      ((:normal)
       (update-all-weights/sgd-normal optimizer))
      ((:nesterov)
       (update-all-weights/sgd-nesterov optimizer))))
  (set-n-instances optimizer gradient-source
                   (+ (n-instances optimizer) n-new-inputs)))

(defun update-all-weights/sgd-no-momentum (optimizer)
  (let ((learning-rate (learning-rate optimizer))
        (n-instances (n-instances-in-batch optimizer))
        (weight-decay (weight-decay optimizer))
        (weight-penalty (weight-penalty optimizer)))
    (cond ((use-segment-derivatives-p optimizer)
           (do-segment-set (segment start-in-segment-set)
                           (segment-set optimizer)
             (declare (ignore start-in-segment-set))
             (let ((accumulator (segment-derivatives segment)))
               (scal! (/ n-instances) accumulator)))
           (map nil #'funcall (before-update-hook optimizer))
           (do-segment-set (segment start-in-segment-set)
                           (segment-set optimizer)
             (declare (ignore start-in-segment-set))
             (let ((weights (segment-weights segment))
                   (accumulator (segment-derivatives segment)))
               (unless (zerop weight-penalty)
                 (add-sign! weight-penalty weights 1 accumulator))
               (unless (zerop weight-decay)
                 (axpy! weight-decay weights accumulator))
               (axpy! (- learning-rate) accumulator weights))))
          (t
           (let ((accumulator (accumulator optimizer)))
             (scal! (/ n-instances) accumulator)
             (map nil #'funcall (before-update-hook optimizer))
             (with-shape-and-displacement (accumulator)
               (do-segment-set (segment start-in-segment-set)
                               (segment-set optimizer)
                 (let ((weights (segment-weights segment)))
                   (reshape-and-displace! accumulator (mat-size weights)
                                          start-in-segment-set)
                   (unless (zerop weight-penalty)
                     (add-sign! weight-penalty weights 1 accumulator))
                   (unless (zerop weight-decay)
                     (axpy! weight-decay weights accumulator))
                   (axpy! (- learning-rate) accumulator weights))))
             (fill! 0 accumulator)))))
  (setf (n-instances-in-batch optimizer) 0)
  (map nil #'funcall (after-update-hook optimizer)))

(defun update-all-weights/sgd-normal (optimizer)
  (assert (not (use-segment-derivatives-p optimizer)) ()
          "SGD-OPTIMIZER supports USE-SEGMENT-DERIVATIVES-P only with
          MOMENTUM :NONE.")
  (let ((accumulator (accumulator optimizer))
        (weight-deltas (weight-deltas optimizer))
        (learning-rate (learning-rate optimizer))
        (n-instances (n-instances-in-batch optimizer))
        (momentum (momentum optimizer))
        (weight-decay (weight-decay optimizer))
        (weight-penalty (weight-penalty optimizer)))
    (assert (and (<= 0 momentum) (< momentum 1)))
    (scal! momentum weight-deltas)
    (cond ((before-update-hook optimizer)
           (scal! (/ n-instances) accumulator)
           (map nil #'funcall (before-update-hook optimizer))
           (axpy! 1 accumulator weight-deltas))
          (t
           (axpy! (/ n-instances) accumulator weight-deltas)))
    (with-shape-and-displacement (weight-deltas)
      (do-segment-set (segment start-in-segment-set) (segment-set optimizer)
        (let ((weights (segment-weights segment)))
          (reshape-and-displace! weight-deltas (mat-size weights)
                                 start-in-segment-set)
          (unless (zerop weight-penalty)
            (add-sign! weight-penalty weights 1 weight-deltas))
          (unless (zerop weight-decay)
            (axpy! weight-decay weights weight-deltas))
          (axpy! (- learning-rate) weight-deltas weights))))
    (fill! 0 accumulator)
    (setf (n-instances-in-batch optimizer) 0))
  (map nil #'funcall (after-update-hook optimizer)))

(defun update-all-weights/sgd-nesterov (optimizer)
  (assert (not (use-segment-derivatives-p optimizer)) ()
          "SGD-OPTIMIZER supports USE-SEGMENT-DERIVATIVES-P only with
          MOMENTUM :NONE.")
  (let ((accumulator (accumulator optimizer))
        (weight-deltas (weight-deltas optimizer))
        (learning-rate (learning-rate optimizer))
        (n-instances (n-instances-in-batch optimizer))
        (momentum (momentum optimizer))
        (weight-decay (weight-decay optimizer))
        (weight-penalty (weight-penalty optimizer)))
    (assert (and (<= 0 momentum) (< momentum 1)))
    (scal! momentum weight-deltas)
    (cond ((before-update-hook optimizer)
           (scal! (/ n-instances) accumulator)
           (map nil #'funcall (before-update-hook optimizer))
           (axpy! 1 accumulator weight-deltas))
          (t
           (axpy! (/ n-instances) accumulator weight-deltas)))
    (with-shape-and-displacement (weight-deltas)
      (with-shape-and-displacement (accumulator)
        (do-segment-set (segment start-in-segment-set) (segment-set optimizer)
          (let ((weights (segment-weights segment)))
            (reshape-and-displace! weight-deltas (mat-size weights)
                                   start-in-segment-set)
            (reshape-and-displace! accumulator (mat-size weights)
                                   start-in-segment-set)
            (unless (zerop weight-decay)
              (axpy! weight-decay weights weight-deltas)
              (scal! (- 1 (* learning-rate weight-decay)) weights))
            (axpy! (- (/ learning-rate n-instances)) accumulator weights)
            (axpy! (- (* learning-rate momentum)) weight-deltas weights)
            (unless (zerop weight-penalty)
              (add-sign! (* learning-rate weight-penalty) weights 0 accumulator)
              (axpy! 1 accumulator weights))))))
    (fill! 0 accumulator)
    (setf (n-instances-in-batch optimizer) 0))
  (map nil #'funcall (after-update-hook optimizer)))


(defsection @mgl-gd-adam-optimizer (:title "Adam Optimizer")
  (adam-optimizer class)
  (learning-rate (accessor adam-optimizer))
  (mean-decay (accessor adam-optimizer))
  (mean-decay-decay (accessor adam-optimizer))
  (variance-decay (accessor adam-optimizer))
  (variance-adjustment (accessor adam-optimizer)))

;;; FIXEXT: Don't allocate variance and mean estimator matrices if the
;;; corresponding rate is known to be constant 1.
(defclass adam-optimizer (batch-gd-optimizer)
  ((learning-rate
    :initform 0.0002 :accessor learning-rate
    :documentation "Same thing as [LEARNING-RATE][(ACCESSOR
    GD-OPTIMIZER)] but with the default suggested by the Adam paper.")
   (momentum-type :initform :none)
   (mean-decay
    :initform 0.9 :initarg :mean-decay
    :accessor mean-decay
    :documentation "A number between 0 and 1 that determines how fast
    the estimated mean of derivatives is updated. 0 basically gives
    you RMSPROP (if VARIANCE-DECAY is not too large) or AdaGrad (if
    VARIANCE-DECAY is close to 1 and the learning rate is annealed.
    This is $\\beta_1$ in the paper.")
   (mean-decay-decay
    :initform (- 1 10d-8) :initarg :mean-decay-decay
    :accessor mean-decay-decay
    :documentation "A value that should be close to 1. MEAN-DECAY is
    multiplied by this value after each update. This is $\\lambda$ in
    the paper." )
   (variance-decay
    :initform 0.999 :initarg :variance-decay
    :accessor variance-decay
    :documentation "A number between 0 and 1 that determines how fast
    the estimated variance of derivatives is updated. This is
    $\\beta_2$ in the paper.")
   (variance-adjustment
    :initform 10d-8 :initarg :variance-adjustment
    :accessor variance-adjustment
    :documentation "Within the bowels of adam, the estimated mean is
    divided by the square root of the estimated variance (per weight)
    which can lead to numerical problems if the denominator is near
    zero. To avoid this, VARIANCE-ADJUSTMENT, which should be a small
    positive number, is added to the denominator. This is `epsilon` in
    the paper.")
   (mean-estimates :accessor mean-estimates)
   (variance-estimates :accessor variance-estimates)
   (adam-time-step :initform 0 :accessor adam-time-step)
   (momentum :initform :none))
  (:documentation "Adam is a first-order stochasistic gradient descent
  optimizer. It maintains an internal estimation for the mean and raw
  variance of each derivative as exponential moving averages. The step
  it takes is basically `M/(sqrt(V)+E)` where `M` is the estimated
  mean, `V` is the estimated variance, and `E` is a small adjustment
  factor to prevent the gradient from blowing up. See version 5 of the
  [paper](http://arxiv.org/abs/1412.6980) for more.

  Note that using momentum is not supported with Adam. In fact, an
  error is signalled if it's not :NONE.

  See @MGL-GD-BATCH-GD-OPTIMIZER for the description of the various
  options common to all batch based optimizers."))

(define-descriptions (optimizer adam-optimizer :inheritp t)
  (mean-decay (mean-decay optimizer) "~,5E")
  (mean-decay-decay (mean-decay-decay optimizer) "~,5E")
  (effective-mean-decay (effective-mean-decay optimizer) "~,5E")
  (variance-decay (variance-decay optimizer) "~,5E")
  (variance-adjustment (variance-adjustment optimizer) "~,5E"))

(defmethod initialize-optimizer* ((optimizer adam-optimizer) source weights
                                  dataset)
  (when (next-method-p)
    (call-next-method))
  (let ((n-weights (size (segment-set optimizer))))
    (setf (variance-estimates optimizer) (make-mat n-weights))
    ;; Create this one lazily to save memory. It's not needed if
    ;; MEAN-DECAY is always 0.
    (setf (mean-estimates optimizer) nil)))

(defun ensure-mean-estimates (optimizer)
  (or (mean-estimates optimizer)
      (setf (mean-estimates optimizer)
            (make-mat (mat-size (variance-estimates optimizer))))))

(defmethod maybe-update-weights ((optimizer adam-optimizer)
                                 gradient-source n-new-inputs)
  (when (<= (batch-size optimizer)
            (incf (n-instances-in-batch optimizer) n-new-inputs))
    (if (use-segment-derivatives-p optimizer)
        (update-all-weights/adam-use-segment-derivatives optimizer)
        (update-all-weights/adam optimizer)))
  (set-n-instances optimizer gradient-source
                   (+ (n-instances optimizer) n-new-inputs)))

(defun effective-mean-decay (optimizer)
  (* (mean-decay optimizer)
     (expt (mean-decay-decay optimizer)
           ;; The MAX 0 is only to produce sensible output in
           ;; DESCRIBE when N-STEPS is 0.
           (max 0 (1- (adam-time-step optimizer))))))

(defun update-all-weights/adam (optimizer)
  (assert (eq (momentum-type optimizer) :none) ()
          "Momentum is not implemented for ADAM-OPTIMIZER.")
  (incf (adam-time-step optimizer))
  (let* ((mean-decay (mean-decay optimizer))
         (mean-decay* (effective-mean-decay optimizer))
         (rmsprop (= 0 mean-decay))
         (accumulator (accumulator optimizer))
         (mean-estimates (if rmsprop
                             accumulator
                             (ensure-mean-estimates optimizer)))
         (variance-estimates (variance-estimates optimizer))
         (variance-decay (variance-decay optimizer))
         (variance-adjustment (variance-adjustment optimizer))
         (learning-rate (learning-rate optimizer))
         (n-instances (n-instances-in-batch optimizer))
         (weight-decay (weight-decay optimizer))
         (weight-penalty (weight-penalty optimizer))
         (step (adam-time-step optimizer)))
    (scal! (/ n-instances) accumulator)
    (map nil #'funcall (before-update-hook optimizer))
    ;; add weight decay and penalty gradients
    (when (or (not (zerop weight-decay))
              (not (zerop weight-penalty)))
      (with-shape-and-displacement (accumulator)
        (do-segment-set (segment start-in-segment-set) (segment-set optimizer)
          (let ((weights (segment-weights segment)))
            (reshape-and-displace! accumulator (mat-size weights)
                                   start-in-segment-set)
            (unless (zerop weight-penalty)
              (add-sign! weight-penalty weights 1 accumulator))
            (unless (zerop weight-decay)
              (axpy! weight-decay weights accumulator))))))
    (unless rmsprop
      (scal! mean-decay* mean-estimates)
      (axpy! (- 1 mean-decay*) accumulator mean-estimates))
    (geem! (- 1 variance-decay) accumulator accumulator
           variance-decay variance-estimates)
    (let ((x (sqrt (- 1 (expt variance-decay step)))))
      (adam-update (* learning-rate (/ x (- 1 (expt mean-decay step))))
                   mean-estimates variance-estimates
                   (* x variance-adjustment) accumulator))
    (with-shape-and-displacement (accumulator)
      (do-segment-set (segment start-in-segment-set) (segment-set optimizer)
        (let ((weights (segment-weights segment)))
          (reshape-and-displace! accumulator (mat-size weights)
                                 start-in-segment-set)
          (axpy! -1 accumulator weights))))
    (fill! 0 accumulator)
    (setf (n-instances-in-batch optimizer) 0))
  (map nil #'funcall (after-update-hook optimizer)))

(defun update-all-weights/adam-use-segment-derivatives (optimizer)
  (assert (eq (momentum-type optimizer) :none) ()
          "Momentum is not implemented for ADAM-OPTIMIZER.")
  (incf (adam-time-step optimizer))
  (let* ((mean-decay (mean-decay optimizer))
         (mean-decay* (effective-mean-decay optimizer))
         (mean-estimates (if (= 0 mean-decay)
                             nil
                             (ensure-mean-estimates optimizer)))
         (variance-estimates (variance-estimates optimizer))
         (variance-decay (variance-decay optimizer))
         (variance-adjustment (variance-adjustment optimizer))
         (learning-rate (learning-rate optimizer))
         (n-instances (n-instances-in-batch optimizer))
         (weight-decay (weight-decay optimizer))
         (weight-penalty (weight-penalty optimizer))
         (step (adam-time-step optimizer)))
    (do-segment-set (segment start-in-segment-set)
                    (segment-set optimizer)
      (declare (ignore start-in-segment-set))
      (let ((accumulator (segment-derivatives segment)))
        (scal! (/ n-instances) accumulator)))
    (map nil #'funcall (before-update-hook optimizer))
    (do-segment-set (segment start-in-segment-set)
                    (segment-set optimizer)
      (let ((weights (segment-weights segment))
            (accumulator (segment-derivatives segment)))
        (unless (zerop weight-penalty)
          (add-sign! weight-penalty weights 1 accumulator))
        (unless (zerop weight-decay)
          (axpy! weight-decay weights accumulator))
        (flet ((foo (mean-estimates)
                 (with-shape-and-displacement (variance-estimates
                                               (mat-dimensions accumulator)
                                               start-in-segment-set)
                   (geem! (- 1 variance-decay) accumulator accumulator
                          variance-decay variance-estimates)
                   (let ((x (sqrt (- 1 (expt variance-decay step)))))
                     (adam-update (* learning-rate
                                     (/ x (- 1 (expt mean-decay step))))
                                  mean-estimates variance-estimates
                                  (* x variance-adjustment) accumulator)))))
          (if mean-estimates
              (with-shape-and-displacement
                  (mean-estimates (mat-dimensions accumulator)
                   start-in-segment-set)
                (scal! mean-decay mean-estimates)
                (axpy! (- 1 mean-decay*) accumulator mean-estimates)
                (foo mean-estimates))
              (foo accumulator)))
        (axpy! -1 accumulator weights)
        (fill! 0 accumulator)))
    (setf (n-instances-in-batch optimizer) 0))
  (map nil #'funcall (after-update-hook optimizer)))

(defun adam-update (step-size mean-estimates variance-estimates
                    variance-adjustment weight-deltas)
  (let* ((n (mat-size mean-estimates))
         (ctype (mat-ctype mean-estimates))
         (step-size (coerce-to-ctype step-size :ctype ctype))
         (variance-adjustment (coerce-to-ctype variance-adjustment
                                               :ctype ctype)))
    (assert (= (mat-size mean-estimates)
               (mat-size variance-estimates)
               (mat-size weight-deltas)))
    (if (use-cuda-p mean-estimates variance-estimates weight-deltas)
        (multiple-value-bind (block-dim grid-dim) (choose-1d-block-and-grid n 4)
          (cuda-adam-update step-size mean-estimates variance-estimates
                            variance-adjustment weight-deltas n
                            :grid-dim grid-dim :block-dim block-dim))
        (progn
          (assert (= 0 (mat-displacement mean-estimates)
                     (mat-displacement variance-estimates)
                     (mat-displacement weight-deltas)))
          (lisp-adam-update step-size mean-estimates variance-estimates
                            variance-adjustment weight-deltas
                            (mat-size mean-estimates))))))

(define-lisp-kernel (lisp-adam-update)
    ((step-size single-float) (mean-estimates :mat :input)
     (variance-estimates :mat :input) (variance-adjustment single-float)
     (weight-deltas :mat :output) (n index))
  (loop for i of-type index below n
        do (setf (aref weight-deltas i)
                 (/ (* step-size (aref mean-estimates i))
                    (+ (the! single-float (sqrt (aref variance-estimates i)))
                       variance-adjustment)))))

(define-cuda-kernel (cuda-adam-update)
    (cl-cuda:void ((step-size float) (mean-estimates :mat :input)
                   (variance-estimates :mat :input) (variance-adjustment float)
                   (weight-deltas :mat :output) (n cl-cuda:int)))
  (let ((stride (* cl-cuda:block-dim-x cl-cuda:grid-dim-x)))
    (do ((i (+ (* cl-cuda:block-dim-x cl-cuda:block-idx-x) cl-cuda:thread-idx-x)
            (+ i stride)))
        ((>= i n))
      (set (aref weight-deltas i) (/ (* step-size (aref mean-estimates i))
                                     (+ (sqrt (aref variance-estimates i))
                                        variance-adjustment))))))


(defsection @mgl-gd-normalized-batch-gd-optimizer
    (:title "Normalized Batch Optimizer")
  (normalized-batch-gd-optimizer class)
  (n-weight-uses-in-batch (accessor normalized-batch-gd-optimizer)))

(defclass normalized-batch-gd-optimizer (batch-gd-optimizer)
  ((n-weight-uses-in-batch
    :accessor n-weight-uses-in-batch
    :documentation "Number of uses of the weight in its current batch."))
  (:documentation "Like BATCH-GD-OPTIMIZER but keeps count of how many
  times each weight was used in the batch and divides the accumulated
  gradient by this count instead of dividing by N-INSTANCES-IN-BATCH.
  This only makes a difference if there are missing values in the
  learner that's being trained. The main feature that distuinguishes
  this class from PER-WEIGHT-BATCH-GD-OPTIMIZER is that batches end at
  same time for all weights."))

(defun set-up-n-weight-uses (optimizer)
  (let ((n-weights (size (segment-set optimizer))))
    (setf (n-weight-uses-in-batch optimizer)
          (make-array n-weights :element-type 'index :initial-element 0))))

(defmethod initialize-optimizer* ((optimizer normalized-batch-gd-optimizer)
                                  source weights dataset)
  (call-next-method)
  (set-up-n-weight-uses optimizer))

(defmethod n-instances-until-update ((optimizer normalized-batch-gd-optimizer))
  ;; Weights are updated as in BATCH-GD-OPTIMIZER but we need to collect
  ;; weight usage statistics after each example.
  1)

(defmethod maybe-update-weights ((optimizer normalized-batch-gd-optimizer)
                                 gradient-source n-new-inputs)
  (declare (type index n-new-inputs))
  (assert (eq (momentum-type optimizer) :normal))
  (assert (not (use-segment-derivatives-p optimizer)) ()
          "NORMALIZED-BATCH-GD-OPTIMIZER does not support ~
          USE-SEGMENT-DERIVATIVES-P.")
  (let ((accumulator (accumulator optimizer))
        (n-weight-uses-in-batch (n-weight-uses-in-batch optimizer))
        (weight-deltas (weight-deltas optimizer))
        (learning-rate (learning-rate optimizer))
        (momentum (momentum optimizer))
        (weight-decay (weight-decay optimizer))
        (weight-penalty (weight-penalty optimizer))
        (batch-size (batch-size optimizer)))
    (declare (type index-vector n-weight-uses-in-batch)
             (type real learning-rate momentum weight-decay weight-penalty)
             (type index batch-size))
    (with-facets ((accumulator (accumulator 'array :direction :io))
                  (weight-deltas (weight-deltas 'array :direction :io)))
      (do-segment-set (segment start-in-segment-set) (segment-set optimizer)
        (map-segment-runs
         (lambda (start end)
           (declare (type index start end)
                    (optimize (speed 3) #.*no-array-bounds-check*))
           (do ((i (the! index (+ start-in-segment-set start))
                   (the! index (1+ i)))
                (j start (1+ j)))
               ((<= end j))
             (setf (aref n-weight-uses-in-batch i)
                   (the! index
                         (+ n-new-inputs
                            (the! index
                                  (aref n-weight-uses-in-batch i)))))))
         segment))
      (when (<= batch-size (the index (incf (n-instances-in-batch optimizer)
                                            n-new-inputs)))
        (setf (n-instances-in-batch optimizer) 0)
        (do-segment-set (segment start-in-segment-set) (segment-set optimizer)
          (let* ((weights (segment-weights segment))
                 (start (mat-displacement weights))
                 (end (the! index (+ start (the index (mat-size weights))))))
            (declare #+nil (optimize (speed 3) #.*no-array-bounds-check*)
                     (type index start end))
            (with-facets ((weights (weights 'backing-array :direction :io)))
              (do ((i start-in-segment-set (the! index (1+ i)))
                   (j start (1+ j)))
                  ((<= end j))
                (let ((delta (+ (* momentum (aref weight-deltas i))
                                (* (if (zerop (aref n-weight-uses-in-batch i))
                                       0
                                       (/ (aref n-weight-uses-in-batch i)))
                                   (aref accumulator i))
                                (* weight-decay (aref weights j))
                                weight-penalty)))
                  (setf (aref weight-deltas i) delta)
                  (decf (aref weights j) (* learning-rate delta))
                  (setf (aref n-weight-uses-in-batch i) 0
                        (aref accumulator i) 0))))))
        (map nil #'funcall (after-update-hook optimizer)))))
  (set-n-instances optimizer gradient-source
                   (+ (n-instances optimizer) n-new-inputs)))


(defsection @mgl-gd-per-weight-optimization (:title "Per-weight Optimization")
  (per-weight-batch-gd-optimizer class)
  (n-weight-uses-in-batch (accessor per-weight-batch-gd-optimizer)))

(defclass per-weight-batch-gd-optimizer (gd-optimizer)
  ((n-weight-uses-in-batch
    :accessor n-weight-uses-in-batch
    :documentation "Number of uses of the weight in its current batch."))
  (:documentation "This is much like @MGL-GD-BATCH-GD-OPTIMIZER but it
  is more clever about when to update weights. Basically every weight
  has its own batch independent from the batches of others. This has
  desirable properties. One can for example put two neural networks
  together without adding any connections between them and the
  learning will produce results equivalent to the separated case.
  Also, adding inputs with only missing values does not change
  anything.

  Due to its very non-batch nature, there is no CUDA implementation of
  this optimizer."))

(defmethod initialize-optimizer* ((optimizer per-weight-batch-gd-optimizer)
                                  source weights dataset)
  (call-next-method)
  (set-up-n-weight-uses optimizer))

(defmethod n-instances-until-update ((optimizer per-weight-batch-gd-optimizer))
  ;; Weight updates are async, don't overpromise.
  1)

(defmethod maybe-update-weights ((optimizer per-weight-batch-gd-optimizer)
                                 gradient-source n-new-inputs)
  (assert (= 1 n-new-inputs))
  (assert (eq (momentum-type optimizer) :normal))
  (assert (not (use-segment-derivatives-p optimizer)) ()
          "PER-WEIGHT-BATCH-GD-OPTIMIZER does not support ~
          USE-SEGMENT-DERIVATIVES-P.")
  (let ((accumulator (accumulator optimizer))
        (n-weight-uses-in-batch (n-weight-uses-in-batch optimizer))
        (weight-deltas (weight-deltas optimizer))
        (learning-rate (learning-rate optimizer))
        (momentum (momentum optimizer))
        (weight-decay (weight-decay optimizer))
        (weight-penalty (weight-penalty optimizer))
        (batch-size (batch-size optimizer)))
    (declare (type index-vector n-weight-uses-in-batch)
             (type real learning-rate momentum weight-decay weight-penalty)
             (type index batch-size))
    (with-facets ((accumulator (accumulator 'array :direction :io))
                  (weight-deltas (weight-deltas 'array :direction :io)))
      #+nil (declare (optimize (speed 3) #.*no-array-bounds-check*))
      (do-segment-set (segment start-in-segment-set) (segment-set optimizer)
        (let ((weights (segment-weights segment)))
          (with-facets ((weights (weights 'array :direction :io)))
            (map-segment-runs
             (lambda (start end)
               (declare (type index start end))
               (do ((i (the! index (+ start-in-segment-set start))
                       (the! index (1+ i)))
                    (j start (1+ j)))
                   ((<= end j))
                 (when (<= batch-size
                           (setf (aref n-weight-uses-in-batch i)
                                 (1+ (the! index
                                           (aref n-weight-uses-in-batch i)))))
                   (let ((delta (+ (* momentum (aref weight-deltas i))
                                   (/ (aref accumulator i)
                                      (aref n-weight-uses-in-batch i))
                                   (* weight-decay (aref weights j))
                                   weight-penalty)))
                     (setf (aref weight-deltas i) delta)
                     (decf (aref weights j) (* learning-rate delta))
                     (setf (aref n-weight-uses-in-batch i) 0
                           (aref accumulator i) (* 0 (aref accumulator i)))))))
             segment)))))
    (map nil #'funcall (after-update-hook optimizer)))
  (set-n-instances optimizer gradient-source
                   (+ (n-instances optimizer) n-new-inputs)))


(defsection @mgl-gd-segmented-gd-optimizer (:title "Segmented GD Optimizer")
  (segmented-gd-optimizer class)
  (segmenter (reader segmented-gd-optimizer))
  (segments (reader segmented-gd-optimizer))
  "SEGMENTED-GD-OPTIMIZER inherits from `ITERATIVE-OPTIMIZER`, so see
  @MGL-OPT-ITERATIVE-OPTIMIZER too.")

(defclass segmented-gd-optimizer (base-gd-optimizer)
  ((segmenter
    :initarg :segmenter :reader segmenter
    :documentation "When this optimizer is initialized it loops over
    the segment of the learner with MAP-SEGMENTS. SEGMENTER is a
    function that is called with each segment and returns an optimizer
    or NIL. Several segments may be mapped to the same optimizer.
    After the segment->optimizer mappings are collected, each
    optimizer is initialized by INITIALIZE-OPTIMIZER with the list of
    segments mapped to it.")
   (optimizers :type list :reader optimizers)
   (segments :type list :reader segments))
  (:documentation "An optimizer that delegates training of segments to
  other optimizers. Useful to delegate training of different segments
  to different optimizers (capable of working with segmentables) or
  simply to not train all segments."))

(define-descriptions (optimizer segmented-gd-optimizer)
  n-instances optimizers segments)

(defmethod describe-object :after ((optimizer segmented-gd-optimizer) stream)
  (when (slot-boundp optimizer 'optimizers)
    (terpri stream)
    (dolist (optimizer (optimizers optimizer))
      (describe optimizer stream))))

(defmethod initialize-optimizer* ((optimizer segmented-gd-optimizer) source
                                  weights dataset)
  (when (next-method-p)
    (call-next-method))
  (let ((segmenter (segmenter optimizer))
        (optimizer-segments (make-hash-table :test 'eq)))
    (map nil (lambda (segment)
               (let ((optimizer (funcall segmenter segment)))
                 (when optimizer
                   (unless (gethash optimizer optimizer-segments)
                     (setf (gethash optimizer optimizer-segments)
                           nil))
                   (push segment (gethash optimizer optimizer-segments)))))
         weights)
    (let ((optimizers ()))
      (maphash (lambda (optimizer segments)
                 (initialize-optimizer* optimizer source segments dataset)
                 (push optimizer optimizers)
                 (values))
               optimizer-segments)
      (setf (slot-value optimizer 'optimizers) optimizers)
      ;; The child optimizer may not use all the segments assigned to it
      ;; so let's ask it.
      (setf (slot-value optimizer 'segments)
            (apply #'append (mapcar #'segments optimizers))))))

(defmethod maybe-update-weights ((optimizer segmented-gd-optimizer)
                                 gradient-source n-new-inputs)
  (dolist (optimizer (optimizers optimizer))
    (maybe-update-weights optimizer gradient-source n-new-inputs))
  (set-n-instances optimizer gradient-source
                   (+ (n-instances optimizer) n-new-inputs)))

(defmethod n-instances-until-update ((optimizer segmented-gd-optimizer))
  (if (optimizers optimizer)
      (loop for child-optimizer in (optimizers optimizer)
            minimizing (n-instances-until-update child-optimizer))
      nil))

(defmethod map-gradient-sink (fn (optimizer segmented-gd-optimizer))
  (dolist (optimizer (optimizers optimizer))
    (map-gradient-sink fn optimizer)))


(defsection @mgl-gd-utilities (:title "Utilities")
  (clip-l2-norm function)
  (arrange-for-clipping-gradients function))

(defun clip-l2-norm (mats l2-upper-bound &key callback)
  "Scale MATS so that their $L_2$ norm does not exceed L2-UPPER-BOUND.

  Compute the norm of of MATS as if they were a single vector. If the
  norm is greater than L2-UPPER-BOUND, then scale each matrix
  destructively by the norm divided by L2-UPPER-BOUND and if non-NIL
  call the function CALLBACK with the scaling factor."
  (let ((sum 0))
    (map nil (lambda (mat)
               (incf sum (expt (nrm2 mat) 2)))
         mats)
    (let ((norm (sqrt sum)))
      (when (< l2-upper-bound norm)
        (let ((scale (/ l2-upper-bound norm)))
          (when callback
            (funcall callback scale))
          (map nil (lambda (mat)
                     (scal! scale mat))
               mats)
          scale)))))

(defun arrange-for-clipping-gradients (batch-gd-optimizer l2-upper-bound
                                       &key callback)
  "Make it so that the norm of the batch normalized gradients
  accumulated by BATCH-GD-OPTIMIZER is clipped to L2-UPPER-BOUND
  before every update. See CLIP-L2-NORM."
  (push (lambda ()
          (clip-l2-norm
           (if (use-segment-derivatives-p batch-gd-optimizer)
               (let ((accumulators ()))
                 (do-gradient-sink ((segment accumulator) batch-gd-optimizer)
                   (declare (ignore segment))
                   (push accumulator accumulators))
                 accumulators)
               (list (mgl-gd::accumulator batch-gd-optimizer)))
           l2-upper-bound :callback callback))
        (before-update-hook batch-gd-optimizer))
  batch-gd-optimizer)

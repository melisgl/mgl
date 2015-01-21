(in-package :mgl-opt)

(defsection @mgl-opt (:title "Gradient Based Optimization")
  "We have a real valued, differentiable function F and the task is to
  find the parameters that minimize its value. Optimization starts
  from a single point in the parameter space of F, and this single
  point is updated iteratively based on the gradient and value of F at
  or around the current point.

  Note that while the stated problem is that of global optimization,
  for non-convex functions, most algorithms will tend to converge to a
  local optimum.

  Currently, there are two optimization algorithms:
  MGL-GD:@MGL-GD (with several variants) and MGL-CG:@MGL-CG both of
  which are first order methods (they do not need second order
  gradients) but more can be added with the @MGL-OPT-EXTENSION-API."
  (minimize function)
  (@mgl-opt-iterative-optimizer section)
  (@mgl-opt-cost section)
  (mgl-gd:@mgl-gd section)
  (mgl-cg:@mgl-cg section)
  (@mgl-opt-extension-api section))

(defun minimize (optimizer gradient-source
                 &key (weights (list-segments gradient-source))
                 (dataset *infinitely-empty-dataset*))
  "Minimize the value of the real valued function represented by
  GRADIENT-SOURCE by updating some of its parameters in WEIGHTS (a MAT
  or a sequence of MATs). Return WEIGHTS. DATASET (see
  MGL:@MGL-DATASET) is a set of unoptimized parameters of the same
  function. For example, WEIGHTS may be the weights of a neural
  network while DATASET is the training set consisting of inputs
  suitable for SET-INPUT. The default
  DATASET, (*INFINITELY-EMPTY-DATASET*) is suitable for when all
  parameters are optimized, so there is nothing left to come from the
  environment.

  Optimization terminates if DATASET is a sampler and it runs out or
  when some other condition met (see TERMINATION, for example). If
  DATASET is a SEQUENCE, then it is reused over and over again.

  Examples for various optimizers are provided in MGL-GD:@MGL-GD and
  MGL-CG:@MGL-CG."
  (let ((weights (ensure-seq weights)))
    (initialize-optimizer* optimizer gradient-source weights dataset)
    (initialize-gradient-source* optimizer gradient-source weights dataset)
    (minimize* optimizer gradient-source weights dataset))
  weights)

(defun ensure-seq (obj)
  (if (typep obj 'sequence)
      obj
      (list obj)))

(defgeneric minimize* (optimizer gradient-source weights dataset)
  (:documentation "Called by MINIMIZE after INITIALIZE-OPTIMIZER* and
  INITIALIZE-GRADIENT-SOURCE*, this generic function is the main
  extension point for writing optimizers."))


(defsection @mgl-opt-iterative-optimizer (:title "Iterative Optimizer")
  (iterative-optimizer class)
  (n-instances (reader iterative-optimizer))
  (termination (accessor iterative-optimizer))
  (on-optimization-started (accessor iterative-optimizer))
  (on-optimization-finished (accessor iterative-optimizer))
  (on-n-instances-changed (accessor iterative-optimizer))
  "Now let's discuss a few handy utilities."
  (monitor-optimization-periodically function)
  (reset-optimization-monitors generic-function)
  (reset-optimization-monitors (method () (iterative-optimizer t)))
  (report-optimization-parameters generic-function))

(defclass iterative-optimizer ()
  ((n-instances
    :initform 0 :initarg :n-instances :reader n-instances
    :documentation "The number of instances this optimizer has seen so
    far. Incremented automatically during optimization.")
   (termination
    :initform nil
    :initarg :termination
    :accessor termination
    :documentation "If a number, it's the number of instances to train
    on in the sense of N-INSTANCES. If N-INSTANCES is equal or greater
    than this value optimization stops. If TERMINATION is NIL, then
    optimization will continue. If it is T, then optimization will
    stop. If it is a function of no arguments, then its return value
    is processed as if it was returned by TERMINATION.")
   (on-optimization-started
    :initform ()
    :initarg :on-optimization-started
    :accessor on-optimization-started
    :documentation "An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called after initializations are
    performed (INITIALIZE-OPTIMIZER*, INITIALIZE-GRADIENT-SOURCE*) but
    before optimization is started.")
   (on-optimization-finished
    :initform ()
    :initarg :on-optimization-finished
    :accessor on-optimization-finished
    :documentation "An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called when optimization has
    finished.")
   (on-n-instances-changed
    :initform ()
    :initarg :on-n-instances-changed
    :accessor on-n-instances-changed
    :documentation "An event hook with parameters `(OPTIMIZER
    GRADIENT-SOURCE N-INSTANCES)`. Called when optimization of a batch
    of instances is done and N-INSTANCES is incremented."))
  (:documentation "An abstract base class of MGL-GD:@MGL-GD and
  MGL-CG:@MGL-CG based optimizers that iterate over instances until a
  termination condition is met."))

(defmethod minimize* :around ((optimizer iterative-optimizer) gradient-source
                              weights dataset)
  (apply-monitors (on-optimization-started optimizer)
                  optimizer gradient-source (n-instances optimizer))
  (multiple-value-prog1
      (call-next-method)
    (apply-monitors (on-optimization-finished optimizer)
                    optimizer gradient-source (n-instances optimizer))))

(defmethod monitors ((optimizer iterative-optimizer))
  ())

(defun monitor-optimization-periodically (optimizer periodic-fns)
  "For each periodic function in the list of PERIODIC-FNS, add a
  monitor to OPTIMIZER's ON-OPTIMIZATION-STARTED,
  ON-OPTIMIZATION-FINISHED and ON-N-INSTANCES-CHANGED hooks. The
  monitors are simple functions that just call each periodic function
  with the event parameters (OPTIMIZER GRADIENT-SOURCE N-INSTANCES).
  Return OPTIMIZER.

  To log and reset the monitors of the gradient source after every
  1000 instances seen by OPTIMIZER:

      (monitor-optimization-periodically optimizer
                                         '((:fn log-my-test-error
                                            :period 2000)
                                           (:fn reset-optimization-monitors
                                            :period 1000
                                            :last-eval 0)))

  Note how we don't pass it's allowed to just pass the initargs for a
  PERIODIC-FN instead of PERIODIC-FN itself. The :LAST-EVAL 0 bit
  prevents RESET-OPTIMIZATION-MONITORS from being called at the start
  of the optimization when the monitors are empty anyway."
  (dolist (periodic-fn periodic-fns)
    (monitor-optimization-periodically* optimizer periodic-fn))
  optimizer)

(defun monitor-optimization-periodically* (optimizer periodic-fn)
  (check-type periodic-fn (or periodic-fn list))
  (let ((periodic-fn (if (listp periodic-fn)
                         (apply #'make-instance 'periodic-fn
                                periodic-fn)
                         periodic-fn)))
    (push (lambda (optimizer gradient-source n-instances)
            (call-periodic-fn! n-instances periodic-fn
                               optimizer gradient-source))
          (on-optimization-started optimizer))
    (push (lambda (optimizer gradient-source n-instances)
            (call-periodic-fn n-instances periodic-fn
                              optimizer gradient-source))
          (on-n-instances-changed optimizer))
    (push (lambda (optimizer gradient-source n-instances)
            (call-periodic-fn! n-instances periodic-fn
                               optimizer gradient-source))
          (on-optimization-finished optimizer))))

(defgeneric reset-optimization-monitors (optimizer gradient-source)
  (:documentation "Report the state of [MONITORS][generic-function] of
  OPTIMIZER and GRADIENT-SOURCE and reset their counters. See
  MONITOR-OPTIMIZATION-PERIODICALLY for an example of how this is
  used."))

(defmethod reset-optimization-monitors ((optimizer iterative-optimizer)
                                        gradient-source)
  "Log the counters of the monitors and reset them."
  (log-msg "training at n-instances: ~S~%"  (n-instances optimizer))
  (let ((counters (remove nil (mapcar #'counter
                                      (append (monitors optimizer)
                                              (monitors gradient-source))))))
    (log-padded counters)
    (map nil #'reset-counter counters)))

(defgeneric report-optimization-parameters (optimizer gradient-source)
  (:documentation "A utility that's often called at the start of
  optimization (from ON-OPTIMIZATION-STARTED). The default
  implementation logs the description of GRADIENT-SOURCE (as in
  DESCRIBE) and OPTIMIZER and calls LOG-CUDA.")
  (:method (optimizer gradient-source)
    (let ((*print-level* nil))
      (with-logging-entry (stream)
        (format stream "Describing gradient source:~%")
        (describe gradient-source stream))
      (with-logging-entry (stream)
        (format stream "Describing optimizer:~%")
        (describe optimizer stream)))
    (log-cuda)))


(defsection @mgl-opt-cost (:title "Cost Function")
  "The function being minimized is often called the _cost_ or the
  _loss_ function."
  (cost generic-function)
  (make-cost-monitors function)
  (make-cost-monitors* generic-function))

(defgeneric cost (model)
  (:documentation "Return the value of the cost function being
  minimized. Calling this only makes sense in the context of an
  ongoing optimization (see MINIMIZE). The cost is that of a batch of
  instances."))

;;; FIXME/FIXDOC: composite models may produce many monitors (i.e. one
;;; per less clump in an FNN), or one (such as in an RNN) where the
;;; time steps make it difficult to go the other way easily.
(defun make-cost-monitors (model &key operation-mode attributes)
  "Return a list of MONITOR objects associated with one BASIC-COUNTER
  each. Implemented in terms of MAKE-COST-MONITORS*."
  (make-cost-monitors* model operation-mode attributes))

(defgeneric make-cost-monitors* (model operation-mode attributes)
  (:documentation "Identical to MAKE-COST-MONITORS bar the keywords
  arguments. Specialize this to add to support for new model types.")
  (:method (object operation-mode attributes)
    (when (applies-to-p #'cost object)
      (list
       (make-instance
        'monitor
        :measurer (lambda (instances result)
                    (declare (ignore instances result))
                    (cost object))
        :counter (make-instance
                  'basic-counter
                  :prepend-attributes
                  (append attributes
                          (if (uninterned-symbol-p (name object))
                              ()
                              `(:component ,(name object)())))))))))


(defsection @mgl-opt-extension-api (:title "Extension API")
  (@mgl-opt-optimizer section)
  (@mgl-opt-gradient-source section)
  (@mgl-opt-gradient-sink section))


(defsection @mgl-opt-optimizer (:title "Implementing Optimizers")
  "The following generic functions must be specialized for new
  optimizer types."
  (minimize* generic-function)
  (initialize-optimizer* generic-function)
  (segments generic-function)
  "The rest are just useful for utilities for implementing
  optimizers."
  (terminate-optimization-p function)
  (set-n-instances function)
  (segment-set class)
  (segments (reader segment-set))
  (size (reader segment-set))
  (do-segment-set macro)
  (segment-set<-mat function)
  (segment-set->mat function))

(defgeneric initialize-optimizer* (optimizer gradient-source weights dataset)
  (:documentation "Called automatically before training starts, this
  function sets up OPTIMIZER to be suitable for optimizing
  GRADIENT-SOURCE. It typically creates appropriately sized
  accumulators for the gradients."))

(defgeneric segments (optimizer)
  (:documentation "Several weight matrices known as *segments* can be
  optimized by a single optimizer. This function returns them as a
  list."))

(defun terminate-optimization-p (n-instances termination)
  "Utility function for subclasses of ITERATIVE-OPTIMIZER. It returns
  whether optimization is to be terminated based on N-INSTANCES and
  TERMINATION that are values of the respective accessors of
  ITERATIVE-OPTIMIZER."
  (cond ((numberp termination)
         (<= termination n-instances))
        ((member termination '(nil t))
         termination)
        (t
         (terminate-optimization-p n-instances (funcall termination)))))

(defun set-n-instances (optimizer gradient-source n-instances)
  "Set [N-INSTANCES][(reader iterative-optimizer)] of OPTIMIZER and
  fire ON-N-INSTANCES-CHANGED. ITERATIVE-OPTIMIZER subclasses must
  call this to increment [N-INSTANCES][(reader iterative-optimizer)]."
  (setf (slot-value optimizer 'n-instances) n-instances)
  (apply-monitors (on-n-instances-changed optimizer)
                  optimizer gradient-source n-instances)
  n-instances)

(defclass segment-set ()
  ((segments
    :initarg :segments :reader segments
    :documentation "A list of weight matrices.")
   (start-indices :reader start-indices)
   (size
    :reader size
    :documentation "The sum of the sizes of the weight matrices of
    SEGMENTS."))
  (:documentation "This is a utility class for optimizers that have a
  list of SEGMENTS and (the weights being optimized) is able to copy
  back and forth between those segments and a single MAT (the
  accumulator)."))

(defmethod print-object ((set segment-set) stream)
  (pprint-logical-block (stream ())
    (print-unreadable-object (set stream :type t :identity t)
      (format stream "~A" (segments set))))
  set)

(defmethod initialize-instance :after ((segment-set segment-set)
                                       &key &allow-other-keys)
  (let ((n 0)
        (start-indices '()))
    (dolist (segment (segments segment-set))
      (push n start-indices)
      (incf n (mat-size (segment-weights segment))))
    (setf (slot-value segment-set 'start-indices) (reverse start-indices)
          (slot-value segment-set 'size) n)))

(defmacro do-segment-set ((segment &optional start) segment-set
                          &body body)
  "Iterate over SEGMENTS in SEGMENT-SET. If START is specified, the it
  is bound to the start index of SEGMENT within SEGMENT-SET. The start
  index is the sum of the sizes of previous segments."
  (alexandria:with-gensyms (%segment-set %start-index)
    `(let* ((,%segment-set ,segment-set))
       (loop for ,segment in (segments ,%segment-set)
             ,@(when start
                 (list 'for %start-index 'in
                       (list 'start-indices %segment-set)))
             do (let (,@(when start
                          (list (list start %start-index))))
                  ,@(when start
                      `((declare (type index ,start))))
                  ,@body)))))

(defun segment-set<-mat (segment-set mat)
  "Copy the values of MAT to the weight matrices of SEGMENT-SET as if
  they were concatenated into a single MAT."
  (map-concat (lambda (m mat) (copy! mat m))
              (segments segment-set) mat :key #'segment-weights))

(defun segment-set->mat (segment-set mat)
  "Copy the values of SEGMENT-SET to MAT as if they were concatenated
  into a single MAT."
  (map-concat #'copy! (segments segment-set) mat :key #'segment-weights))


(defsection @mgl-opt-gradient-source (:title "Implementing Gradient Sources")
  "Weights can be stored in a multitude of ways. Optimizers need to
  update weights, so it is assumed that weights are stored in any
  number of MAT objects called segments.

  The generic functions in this section must all be specialized for
  new gradient sources except where noted."
  (map-segments generic-function)
  (map-segment-runs generic-function)
  (segment-weights generic-function)
  (segment-weights (method () (mat)))
  (list-segments function)
  (initialize-gradient-source* generic-function)
  (initialize-gradient-source* (method () (t t t t)))
  (accumulate-gradients* generic-function))

(defgeneric map-segments (fn gradient-source)
  (:documentation "Apply FN to each segment of GRADIENT-SOURCE.")
  (:method (fn (segment-list list))
    (mapc fn segment-list)))

(defgeneric segment-weights (segment)
  (:documentation "Return the weight matrix of SEGMENT. A segment
  doesn't need to be a MAT object itself. For example, it may be a
  MGL-BM:CHUNK of a [MGL-BM:BM][CLASS] or a MGL-BP:LUMP of a
  [MGL-BP:BPN][CLASS] whose NODES slot holds the weights.")
  (:method ((mat mat))
    "When the segment is really a MAT, then just return it."
    mat))

(defgeneric map-segment-runs (fn segment)
  (:documentation "Call FN with start and end of intervals of
  consecutive indices that are not missing in SEGMENT. Called by
  optimizers that support partial updates. The default implementation
  assumes that all weights are present. This only needs to be
  specialized if one plans to use an optimizer that knows how to deal
  unused/missing weights such as MGL-GD:NORMALIZED-BATCH-GD-OPTIMIZER
  and OPTIMIZER MGL-GD:PER-WEIGHT-BATCH-GD-OPTIMIZER.")
  (:method (fn segment)
    (let ((mat (segment-weights segment)))
      (funcall fn mat 0 (mat-size mat)))))

(defun list-segments (gradient-source)
  "A utility function that returns the list of segments from
  MAP-SEGMENTS on GRADIENT-SOURCE."
  (let ((segments ()))
    (map-segments (lambda (segment)
                    (push segment segments))
                  gradient-source)
    (reverse segments)))

(defgeneric initialize-gradient-source* (optimizer gradient-source weights
                                         dataset)
  (:documentation "Called automatically before MINIMIZE* is called,
  this function may be specialized if GRADIENT-SOURCE needs some kind
  of setup.")
  (:method (optimizer gradient-source weights dataset)
    "The default method does nothing."
    nil))

(defgeneric accumulate-gradients* (gradient-source sink batch multiplier valuep)
  (:documentation "Add MULTIPLIER times the sum of first-order
  gradients to accumulators of SINK (normally accessed with
  DO-GRADIENT-SINK) and if VALUEP, return the sum of values of the
  function being optimized for a BATCH of instances. GRADIENT-SOURCE
  is the object representing the function being optimized, SINK is
  gradient sink.

  Note the number of instances in BATCH may be larger than what
  GRADIENT-SOURCE process in one go (in the sense of say,
  MAX-N-STRIPES), so DO-BATCHES-FOR-MODEL or something like (GROUP
  BATCH MAX-N-STRIPES) can be handy."))


(defsection @mgl-opt-gradient-sink (:title "Implementing Gradient Sinks")
  "Optimizers call ACCUMULATE-GRADIENTS* on gradient sources. One
  parameter of ACCUMULATE-GRADIENTS* is the SINK. A gradient sink
  knows what accumulator matrix (if any) belongs to a segment. Sinks
  are defined entirely by MAP-GRADIENT-SINK."
  (map-gradient-sink generic-function)
  (do-gradient-sink macro))

(defgeneric map-gradient-sink (fn sink)
  (:documentation "Call FN of lambda list (SEGMENT ACCUMULATOR) on
  each segment and their corresponding accumulator MAT in SINK."))

(defmacro do-gradient-sink (((segment accumulator) sink)
                            &body body)
  "A convenience macro on top of MAP-GRADIENT-SINK."
  `(map-gradient-sink (lambda (,segment ,accumulator)
                        ,@body)
                      ,sink))

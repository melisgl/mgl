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
  (*accumulating-interesting-gradients* variable)
  (@mgl-opt-iterative-optimizer section)
  (mgl-gd:@mgl-gd section)
  (mgl-cg:@mgl-cg section)
  (@mgl-opt-extension-api section))

(defvar *accumulating-interesting-gradients* nil
  "FIXME: Will go away soon.")

(defun minimize (optimizer gradient-source
                 &key (weights (list-segments gradient-source))
                 (dataset *infinitely-empty-dataset*))
  "Minimize the value of the real valued function represented by
  GRADIENT-SOURCE by updating some of its parameters in WEIGHTS (a MAT
  or a sequence of MATs). Return WEIGHTS. DATASET (see
  MGL:@MGL-DATASETS) is a set of unoptimized parameters of the same
  function. For example, WEIGHTS may be the weights of a neural
  network while DATASET is the training set consisting of inputs
  suitable for MGL-TRAIN:SET-INPUT. The default DATASET,
  (*EMPTY-DATASET*) is suitable for when all parameters are optimized,
  so there is nothing left to come from the environment.

  Optimization terminates if DATASET is a sampler and it runs out or
  when some other condition met (see TERMINATION, for example). If
  DATASET is a SEQUENCE, then it is reused over and over again."
  (let ((weights (ensure-seq weights)))
    (initialize-optimizer* optimizer gradient-source weights dataset)
    (initialize-gradient-source* optimizer gradient-source weights dataset)
    (minimize* optimizer gradient-source weights dataset))
  weights)

(defun ensure-seq (obj)
  (if (typep obj 'sequence)
      obj
      (list obj)))


(defsection @mgl-opt-iterative-optimizer (:title "Iterative Optimizer")
  (iterative-optimizer class)
  (n-instances (reader iterative-optimizer))
  (termination (accessor iterative-optimizer))
  (set-n-instances generic-function))

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
    is processed as if it was returned by TERMINATION."))
  (:documentation "An abstract base class of MGL-GD:@MGL-GD and
  MGL-CG:@MGL-CG based optimizers that iterate over instances until a
  termination condition is met."))

(defgeneric set-n-instances (optimizer gradient-source n-instances)
  (:documentation "Called whenever N-INSTANCES of OPTIMIZER is
  incremented. Hang an :AFTER method on this to print some
  statistics.")
  (:method ((optimizer iterative-optimizer) gradient-source n-instances)
    (setf (slot-value optimizer 'n-instances) n-instances)))


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
  (segment-set class)
  (segments (reader segment-set))
  (size (reader segment-set))
  (do-segment-set macro)
  (segment-set<-mat function)
  (segment-set->mat function))

(defgeneric minimize* (optimizer gradient-source weights dataset))

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
  gradient sink."))


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

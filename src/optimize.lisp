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
  (@mgl-opt-extension-api section)
  (@mgl-opt-iterative-optimizer section)
  (mgl-gd:@mgl-gd section)
  (mgl-cg:@mgl-cg section))

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
  (minimize* generic-function)
  (initialize-optimizer* generic-function)
  (terminate-optimization-p function)
  (segment-set class)
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
  (:documentation "The list of segments optimized by OPTIMIZER."))

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
  ((segments :initform (error "Must specify segment list.")
             :initarg :segments :reader segments)
   (start-indices :reader start-indices)
   (size
    :reader size
    :documentation "The sum of the sizes of the weight matrices of
    SEGMENTS."))
  (:documentation "It's like a concatenation of segments."))

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

(defmacro do-segment-set ((segment &key start-in-segment-set) segment-set
                          &body body)
  "Iterate over SEGMENTS in SEGMENT-SET ...."
  (alexandria:with-gensyms (%segment-set %start-index)
    `(let* ((,%segment-set ,segment-set))
       (loop for ,segment in (segments ,%segment-set)
             ,@(when start-in-segment-set
                 (list 'for %start-index 'in
                       (list 'start-indices %segment-set)))
             do
             (let (,@(when start-in-segment-set
                       (list (list start-in-segment-set %start-index))))
               ,@(when start-in-segment-set
                   `((declare (type index ,start-in-segment-set))))
               ,@body)))))

(defun segment-set<-mat (segment-set mat)
  "Copy the values of MAT to SEGMENT-SET."
  (map-concat (lambda (m mat) (copy! mat m))
              (segments segment-set) mat :key #'segment-weights))

(defun segment-set->mat (segment-set mat)
  "Copy the values of SEGMENT-SET to MAT."
  (map-concat #'copy! (segments segment-set) mat :key #'segment-weights))


(defsection @mgl-opt-gradient-source (:title "Implementing Gradient Sources")
  "Weights can be stored in a multitude of ways. It is assumed that
  weights are stored in any number of MAT objects."
  (map-segments generic-function)
  (list-segments function)
  (initialize-gradient-source* generic-function)
  (accumulate-gradients* generic-function)
  (map-segments generic-function)
  (map-segment-runs generic-function)
  (segment-weights generic-function))

(defgeneric map-segments (fn gradient-source)
  (:documentation "Apply FN to each segment of GRADIENT-SOURCE.")
  (:method (fn (segment-list list))
    (mapc fn segment-list)))

(defgeneric segment-weights (segment)
  (:documentation "Return the weight matrix of SEGMENT.")
  (:method ((mat mat))
    mat))

(defgeneric map-segment-runs (fn segment)
  (:documentation "Call FN with start and end of intervals of
  consecutive indices that are not missing in SEGMENT. Called by
  optimizers that support partial updates.")
  (:method (fn segment)
    (let ((mat (segment-weights segment)))
      (funcall fn mat 0 (mat-size mat)))))

(defun list-segments (gradient-source)
  "Return the list of segments from MAP-SEGMENTS on GRADIENT-SOURCE."
  (let ((segments ()))
    (map-segments (lambda (segment)
                    (push segment segments))
                  gradient-source)
    (reverse segments)))

(defgeneric initialize-gradient-source* (optimizer gradient-source weights
                                         dataset)
  (:documentation "Called automatically before training starts, this
  function sets up SINK to be suitable for SOURCE. It typically
  creates accumulator arrays in the sink for the gradients.")
  (:method (optimizer gradient-source weights dataset)))

(defgeneric accumulate-gradients* (source sink batch multiplier valuep)
  (:documentation "Add MULTIPLIER times the sum of first-order
  gradients to accumulators of SINK (normally accessed with
  DO-GRADIENT-SINK) and if VALUEP, return the sum of values of the
  function being optimized for a BATCH of instances. SOURCE is the
  object representing the function being optimized, SINK is gradient
  sink."))


(defsection @mgl-opt-gradient-sink (:title "Implementing Gradient Sinks")
  (map-gradient-sink generic-function)
  (do-gradient-sink macro)
  (call-with-sink-accumulator generic-function)
  (with-sink-accumulator macro)
  (accumulated-in-sink-p function))

(defgeneric map-gradient-sink (fn sink)
  (:documentation "Call FN of lambda list (SEGMENT ACCUMULATOR) on
  each segment and their corresponding accumulator MAT in SINK."))

(defmacro do-gradient-sink (((segment accumulator) sink)
                            &body body)
  `(map-gradient-sink (lambda (,segment ,accumulator)
                        ,@body)
                      ,sink))

(defgeneric call-with-sink-accumulator (fn segment source sink)
  (:method (fn segment source sink)
    (declare (ignore source))
    (do-gradient-sink ((segment2 accumulator) sink)
      (when (eq segment2 segment)
        (funcall fn accumulator)))))

(defmacro with-sink-accumulator ((accumulator (segment source sink))
                                 &body body)
  `(call-with-sink-accumulator (lambda (,accumulator)
                                 ,@body)
                               ,segment ,source ,sink))

(defun accumulated-in-sink-p (segment source sink)
  (call-with-sink-accumulator (lambda (accumulator)
                                (declare (ignore accumulator))
                                (return-from accumulated-in-sink-p t))
                              segment source sink)
  nil)

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
  (*empty-dataset* variable)
  (*accumulating-interesting-gradients* variable)
  (@mgl-opt-extension-api section)
  (@mgl-opt-iterative-optimizer section)
  (mgl-gd:@mgl-gd section)
  (mgl-cg:@mgl-cg section))

(defvar *empty-dataset* (make-instance 'mgl-train:function-sampler
                                       :sampler (constantly nil))
  "This is the default dataset for MINIMIZE. It's an infinite set of
  stream of NILs.")

(defvar *accumulating-interesting-gradients* nil)

(defun minimize (optimizer gradient-source
                 &key (weights (list-segments gradient-source))
                 (dataset *empty-dataset*))
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
  (terminate-optimization-p function))

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


(defsection @mgl-opt-gradient-source (:title "Implementing Gradient Sources")
  (initialize-gradient-source* generic-function)
  (accumulate-gradients* generic-function))

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

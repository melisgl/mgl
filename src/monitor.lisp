(in-package :mgl-core)

(defsection @mgl-monitoring (:title "Monitoring")
  "When training or applying a model, one often wants to track various
  statistics. For example, in the case of training a neural network
  with cross-entropy loss, these statistics could be the average
  cross-entropy loss itself, classification accuracy, or even the
  entire confusion matrix and sparsity levels in hidden layers. Also,
  there is the question of what to do with the measured values (log
  and forget, add to some counter or a list).

  So there may be several phases of operation when we want to keep an
  eye on. Let's call these **events**. There can also be many fairly
  independent things to do in response to an event. Let's call these
  **monitors**. Some monitors are a composition of two operations: one
  that extracts some measurements and another that aggregates those
  measurements. Let's call these two **measurers** and **counters**,
  respectively.

  For example, consider training a backpropagation neural network. We
  want to look at the state of of network just after the backward
  pass. MGL-BP:BP-LEARNER has a [MONITORS][(accessor
  mgl-bp:bp-learner)] event hook corresponding to the moment after
  backpropagating the gradients. Suppose we are interested in how the
  training cost evolves:

      (push (make-instance 'monitor
                           :measurer (lambda (instances bpn)
                                       (declare (ignore instances))
                                       (mgl-bp:cost bpn))
                           :counter (make-instance 'basic-counter))
            (monitors learner))

  During training, this monitor will track the cost of training
  examples behind the scenes. If we want to print and reset this
  monitor periodically we can put another monitor on
  MGL-OPT:ITERATIVE-OPTIMIZER's MGL-OPT:ON-N-INSTANCES-CHANGED
  accessor:

      (push (lambda (optimizer gradient-source n-instances)
              (declare (ignore optimizer))
              (when (zerop (mod n-instances 1000))
                (format t \"n-instances: ~S~%\" n-instances)
                (dolist (monitor (monitors gradient-source))
                  (when (counter monitor)
                    (format t \"~A~%\" (counter monitor))
                    (reset-counter (counter monitor)))))
            (mgl-opt:on-n-instances-changed optimizer))

  Note that the monitor we push can be anything as long as
  APPLY-MONITOR is implemented on it with the appropriate signature.
  Also note that the ZEROP + MOD logic is fragile, so you will likely
  want to use MGL-OPT:MONITOR-OPTIMIZATION-PERIODICALLY instead of
  doing the above.

  So that's the general idea. Concrete events are documented where
  they are signalled. Often there are task specific utilities that
  create a reasonable set of default monitors (see
  @MGL-CLASSIFICATION-MONITOR)."
  (apply-monitors function)
  (apply-monitor generic-function)
  (counter generic-function)
  (monitor-model-results function)
  (monitors generic-function)
  (@mgl-monitor section)
  (@mgl-measurer section)
  (@mgl-counter section))

(defun apply-monitors (monitors &rest arguments)
  "Call APPLY-MONITOR on each monitor in MONITORS and ARGUMENTS. This
  is how an event is fired."
  (dolist (monitor monitors)
    (apply #'apply-monitor monitor arguments)))

(defgeneric apply-monitor (monitor &rest arguments)
  (:documentation "Apply MONITOR to ARGUMENTS. This sound fairly
  generic, because it is. MONITOR can be anything, even a simple
  function or symbol, in which case this is just CL:APPLY. See
  @MGL-MONITOR for more.")
  (:method ((monitor function) &rest arguments)
    (apply monitor arguments))
  (:method ((monitor symbol) &rest arguments)
    (apply monitor arguments)))

(defgeneric counter (monitor)
  (:documentation "Return an object representing the state of MONITOR
  or NIL, if it doesn't have any (say because it's a simple logging
  function). Most monitors have counters into which they accumulate
  results until they are printed and reset. See @MGL-COUNTER for
  more.")
  (:method ((monitor function))
    nil)
  (:method ((monitor symbol))
    nil))

(defun monitor-model-results (fn dataset model monitors)
  "Call FN with batches of instances from DATASET until it runs
  out (as in DO-BATCHES-FOR-MODEL). FN is supposed to apply MODEL to
  the batch and return some kind of result (for neural networks, the
  result is the model state itself). Apply MONITORS to each batch and
  the result returned by FN for that batch. Finally, return the list
  of counters of MONITORS.

  The purpose of this function is to collect various results and
  statistics (such as error measures) efficiently by applying the
  model only once, leaving extraction of quantities of interest from
  the model's results to MONITORS.

  See the model specific versions of this functions such as
  MGL-BP:MONITOR-BPN-RESULTS."
  (when monitors
    (do-batches-for-model (batch (dataset model))
      (apply-monitors monitors batch (funcall fn batch)))
    (map 'list #'counter monitors)))

(defgeneric monitors (object)
  (:documentation "Return monitors associated with OBJECT. See various
  methods such as [MONITORS][(accessor mgl-bp:bp-learner)] for more
  documentation."))


(defsection @mgl-monitor (:title "Monitors")
  (monitor class)
  (measurer (reader monitor))
  (counter (reader monitor)))

(defclass monitor ()
  ((measurer
    :initarg :measurer :reader measurer
    :documentation "This must be a monitor itself which only means
    that APPLY-MONITOR is defined on it (but see @MGL-MONITORING). The
    returned values are aggregated by [COUNTER][(READER MONITOR)]. See
    @MGL-MEASURER for a library of measurers.")
   (counter
    :initarg :counter :reader counter
    :documentation "The COUNTER of a monitor carries out the
    aggregation of results returned by MEASURER. The See @MGL-COUNTER
    for a library of counters."))
  (:documentation "A monitor that has another monitor called MEASURER
  embedded in it. When this monitor is applied, it applies the
  measurer and passes the returned values to ADD-TO-COUNTER called on
  its COUNTER slot. One may further specialize APPLY-MONITOR to change
  that.

  This class is useful when the same event monitor is applied
  repeatedly over a period and its results must be aggregated such as
  when training statistics are being tracked or when predictions are
  begin made. Note that the monitor must be compatible with the event
  it handles. That is, the embedded MEASURER must be prepared to take
  the arguments that are documented to come with the event."))

(defmethod print-object ((monitor monitor) stream)
  (print-unreadable-object (monitor stream :type t)
    (let ((*print-escape* nil))
      (print-object (counter monitor) stream))))

(defmethod apply-monitor ((monitor monitor) &rest arguments)
  (multiple-value-call #'add-to-counter (counter monitor)
    (apply #'apply-monitor (measurer monitor)
           arguments)))
